from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data classes used by views
# ---------------------------------------------------------------------------

@dataclass
class GAStats:
    """Holds everything about one GA run so the view can explain it clearly."""
    population_size: int
    generations: int
    crossover_prob: float
    mutation_prob: float
    initial_rmse: float
    final_rmse: float
    improvement_pct: float
    best_generation: int
    fitness_history: List[float]   # best RMSE per generation (for chart)


@dataclass
class Recommendation:
    product_id: int
    category: str
    price: float
    score: float
    rank: int
    # Explanation fields built from chromosome weights
    category_weight: float   # GA-learned weight for this product's category
    price_weight: float      # GA-learned weight for price dimension
    dominant_signal: str     # "rating" | "click" | "purchase" | "view"
    reason: str              # Human-readable explanation


@dataclass
class UserPageData:
    """Single object returned to the view — GA runs exactly once."""
    user: dict
    ga_stats: GAStats
    chromosome_weights: Dict[str, float]   # feature_name → weight
    top_categories: List[Tuple[str, float]]
    seen_count: int
    recommendations: List[Recommendation]


# ---------------------------------------------------------------------------
# Main recommender
# ---------------------------------------------------------------------------

class GeneticRecommender:
    """
    GA-based recommender inspired by ETagMF (Almaayah et al., 2025).

    Mapping to the paper:
      - Chromosome  = user-feature weight vector  (analogous to user-tag matrix row)
      - P matrix    = product feature matrix (one-hot category + price)
      - Fitness     = RMSE between predicted and actual user scores
      - Selection   = Roulette Wheel (lower RMSE → higher selection probability)
      - Crossover   = single-point gene swap  (Pc = 0.8)
      - Mutation    = random gene replacement  (Pm = 0.12)
      - Elitism     = top 20% survive each generation unchanged
    """

    def __init__(
        self,
        data_dir: str,
        population_size: int = 50,
        generations: int = 80,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.12,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.rng = np.random.default_rng(seed)

        # Load all data once at startup
        self.users    = pd.read_excel(self.data_dir / "users.xlsx")
        self.products = pd.read_excel(self.data_dir / "products.xlsx")
        self.ratings  = pd.read_excel(self.data_dir / "ratings.xlsx")
        self.behavior = pd.read_excel(self.data_dir / "behavior.xlsx")

        # Ensure consistent int types for join keys
        self.products["product_id"] = self.products["product_id"].astype(int)
        self.users["user_id"]       = self.users["user_id"].astype(int)
        self.ratings[["user_id", "product_id"]]  = self.ratings[["user_id", "product_id"]].astype(int)
        self.behavior[["user_id", "product_id"]] = self.behavior[["user_id", "product_id"]].astype(int)

        # Build static product feature matrix (P matrix in ETagMF terms)
        self.feature_df   = self._build_product_features()
        self.feature_cols = [c for c in self.feature_df.columns if c.startswith("cat_")] + ["price_scaled"]

        # Map product_id → dominant behavior signal (for explanation)
        self._behavior_signal = self._compute_behavior_signals()

    # ------------------------------------------------------------------
    # Feature engineering (product P-matrix)
    # ------------------------------------------------------------------

    def _build_product_features(self) -> pd.DataFrame:
        """
        Build item-feature matrix P.
        Columns: one-hot category flags + normalized price.
        Values are in [0, 1] — same range as the chromosome genes.
        """
        df = self.products.copy()
        cat_ohe = pd.get_dummies(df["category"], prefix="cat").astype(float)
        price_min, price_max = df["price"].min(), df["price"].max()
        df["price_scaled"] = (df["price"] - price_min) / (price_max - price_min + 1e-8)
        return pd.concat([df[["product_id", "category", "price"]], cat_ohe, df[["price_scaled"]]], axis=1)

    def _compute_behavior_signals(self) -> Dict[int, str]:
        """
        For each product × user interaction, decide which signal was strongest.
        Used to build human-readable recommendation reasons.
        """
        b = self.behavior.copy()
        b["dominant"] = "viewed"
        b.loc[b["clicked"]   > 0, "dominant"] = "clicked"
        b.loc[b["purchased"] > 0, "dominant"] = "purchased"

        r = self.ratings[["product_id"]].drop_duplicates().copy()
        r["dominant"] = "rated"

        combined = pd.concat([b[["product_id", "dominant"]], r])
        # Keep the strongest signal per product across all users
        priority = {"purchased": 4, "rated": 3, "clicked": 2, "viewed": 1}
        combined["rank"] = combined["dominant"].map(priority)
        combined = combined.sort_values("rank", ascending=False).drop_duplicates("product_id")
        return dict(zip(combined["product_id"], combined["dominant"]))

    # ------------------------------------------------------------------
    # User training data
    # ------------------------------------------------------------------

    def _build_user_training_data(
        self, user_id: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build (X_train, y_train, seen_product_ids) for one user.

        y_train is a blended score in [0, 1] combining:
          - Explicit rating (weight 0.65): normalized to [0,1]
          - Implicit behavior (weight 0.35): purchased > clicked > viewed
        """
        ratings_u  = self.ratings[self.ratings["user_id"] == user_id][["product_id", "rating"]]
        behavior_u = self.behavior[self.behavior["user_id"] == user_id][
            ["product_id", "viewed", "clicked", "purchased"]
        ]

        merged = (
            self.products[["product_id"]]
            .merge(ratings_u,  on="product_id", how="left")
            .merge(behavior_u, on="product_id", how="left")
            .fillna(0.0)
        )

        rating_score   = merged["rating"] / 5.0
        behavior_score = (
            0.15 * merged["viewed"].clip(upper=1)
            + 0.30 * merged["clicked"].clip(upper=1)
            + 0.55 * merged["purchased"].clip(upper=1)
        )
        blended = np.clip(0.65 * rating_score + 0.35 * behavior_score, 0.0, 1.0)

        has_interaction = (
            (merged["rating"]    > 0)
            | (merged["viewed"]  > 0)
            | (merged["clicked"] > 0)
            | (merged["purchased"] > 0)
        )

        seen_df = merged.loc[has_interaction, ["product_id"]].copy()
        seen_df["target"] = blended[has_interaction].values

        feature_merged = seen_df.merge(
            self.feature_df[["product_id"] + self.feature_cols], on="product_id", how="left"
        )

        x_train       = feature_merged[self.feature_cols].to_numpy(dtype=float)
        y_train       = feature_merged["target"].to_numpy(dtype=float)
        seen_products = seen_df["product_id"].to_numpy(dtype=int)
        return x_train, y_train, seen_products

    # ------------------------------------------------------------------
    # GA core
    # ------------------------------------------------------------------

    def _fitness(self, weights: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        """
        RMSE between predicted scores and actual blended scores.
        Lower is better — matches the fitness function in ETagMF (Equation 4).
        """
        preds = np.clip(x @ weights, 0.0, 1.0)
        rmse  = float(np.sqrt(np.mean((preds - y) ** 2)))
        return rmse

    def _evolve(
        self, x_train: np.ndarray, y_train: np.ndarray
    ) -> Tuple[np.ndarray, GAStats]:
        """
        Run the full GA loop and return (best_chromosome, GAStats).

        Steps (mirror ETagMF Algorithm 1):
          1. Initialize population of Np chromosomes randomly in [0,1]
          2. Evaluate fitness (RMSE) for each chromosome
          3. Sort by fitness; keep elite top 20%
          4. Roulette-wheel select parents; apply crossover + mutation
          5. Replace worst members; repeat for MaxGeneration generations
          6. Return chromosome with lowest RMSE
        """
        dims = x_train.shape[1]

        # Step 1 — random initialization (genes in [0,1] as in ETagMF)
        population = self.rng.uniform(0.0, 1.0, size=(self.population_size, dims))

        fitness_history: List[float] = []
        best_gen        = 0
        best_rmse_ever  = float("inf")

        # Record initial RMSE before any evolution
        init_fitness = np.array([self._fitness(c, x_train, y_train) for c in population])
        initial_rmse = float(init_fitness.min())

        for gen in range(self.generations):
            # Step 2 — evaluate fitness
            fitness_vals = np.array([self._fitness(c, x_train, y_train) for c in population])
            order        = np.argsort(fitness_vals)
            population   = population[order]
            fitness_vals = fitness_vals[order]

            best_rmse_this_gen = float(fitness_vals[0])
            fitness_history.append(round(best_rmse_this_gen, 6))

            if best_rmse_this_gen < best_rmse_ever:
                best_rmse_ever = best_rmse_this_gen
                best_gen       = gen

            # Step 3 — elitism: keep top 20%
            elite_count  = max(2, self.population_size // 5)
            next_pop     = [population[i].copy() for i in range(elite_count)]

            # Roulette wheel: invert fitness so lower RMSE → higher probability
            inv   = 1.0 / (fitness_vals + 1e-8)
            probs = inv / inv.sum()

            # Step 4 — crossover + mutation to fill remaining slots
            while len(next_pop) < self.population_size:
                idx1, idx2 = self.rng.choice(self.population_size, size=2, replace=False, p=probs)
                p1, p2     = population[idx1].copy(), population[idx2].copy()
                c1, c2     = p1.copy(), p2.copy()

                # Single-point crossover (Pc)
                if self.rng.random() < self.crossover_prob:
                    cut      = int(self.rng.integers(1, dims))
                    c1[:cut] = p2[:cut]
                    c2[:cut] = p1[:cut]

                # Bit-flip mutation (Pm per gene)
                for child in (c1, c2):
                    mask         = self.rng.random(dims) < self.mutation_prob
                    child[mask]  = self.rng.uniform(0.0, 1.0, size=mask.sum())
                    next_pop.append(child)
                    if len(next_pop) == self.population_size:
                        break

            population = np.array(next_pop)

        # Step 6 — pick best chromosome
        final_fitness = np.array([self._fitness(c, x_train, y_train) for c in population])
        best_weights  = population[np.argmin(final_fitness)].copy()
        final_rmse    = float(final_fitness.min())

        improvement = ((initial_rmse - final_rmse) / (initial_rmse + 1e-8)) * 100.0

        stats = GAStats(
            population_size  = self.population_size,
            generations      = self.generations,
            crossover_prob   = self.crossover_prob,
            mutation_prob    = self.mutation_prob,
            initial_rmse     = round(initial_rmse, 5),
            final_rmse       = round(final_rmse,   5),
            improvement_pct  = round(max(improvement, 0.0), 2),
            best_generation  = best_gen,
            fitness_history  = fitness_history,
        )
        return best_weights, stats

    # ------------------------------------------------------------------
    # Public API  (single entry point for the view)
    # ------------------------------------------------------------------

    def get_user_page_data(self, user_id: int, top_n: int = 12) -> UserPageData:
        """
        Run GA exactly once and return everything the view needs.
        """
        user_row = self.users[self.users["user_id"] == user_id]
        if user_row.empty:
            raise ValueError(f"User {user_id} not found")

        user = user_row.iloc[0].to_dict()
        x_train, y_train, seen_products = self._build_user_training_data(user_id)

        # Cold-start fallback: uniform weights if no interaction history
        if len(x_train) == 0:
            weights = np.full(len(self.feature_cols), 1.0 / len(self.feature_cols))
            stats   = GAStats(
                population_size = self.population_size,
                generations     = self.generations,
                crossover_prob  = self.crossover_prob,
                mutation_prob   = self.mutation_prob,
                initial_rmse    = 0.0,
                final_rmse      = 0.0,
                improvement_pct = 0.0,
                best_generation = 0,
                fitness_history = [],
            )
        else:
            weights, stats = self._evolve(x_train, y_train)

        # Build chromosome weight map  (feature_name → weight)
        weight_map = {name: round(float(w), 4) for name, w in zip(self.feature_cols, weights)}

        # Extract category weights for profile display
        cat_weights   = {k.replace("cat_", ""): v for k, v in weight_map.items() if k.startswith("cat_")}
        top_categories = sorted(cat_weights.items(), key=lambda kv: kv[1], reverse=True)[:5]

        # Build recommendations (only unseen products)
        all_features = self.feature_df[self.feature_cols].to_numpy(dtype=float)
        all_scores   = np.clip(all_features @ weights, 0.0, 1.0)

        result = self.feature_df.copy()
        result["score"] = all_scores
        if len(seen_products) > 0:
            result = result[~result["product_id"].isin(seen_products)]
        result = result.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

        recommendations: List[Recommendation] = []
        for rank, (_, row) in enumerate(result.iterrows(), start=1):
            cat_key     = f"cat_{row['category']}"
            cat_w       = weight_map.get(cat_key, 0.0)
            price_w     = weight_map.get("price_scaled", 0.0)
            pid         = int(row["product_id"])
            signal      = self._behavior_signal.get(pid, "unrated")
            reason      = self._build_reason(row["category"], cat_w, price_w, signal, float(row["score"]))
            recommendations.append(
                Recommendation(
                    product_id       = pid,
                    category         = str(row["category"]),
                    price            = float(row["price"]),
                    score            = round(float(row["score"]), 4),
                    rank             = rank,
                    category_weight  = round(cat_w, 4),
                    price_weight     = round(price_w, 4),
                    dominant_signal  = signal,
                    reason           = reason,
                )
            )

        return UserPageData(
            user             = user,
            ga_stats         = stats,
            chromosome_weights = weight_map,
            top_categories   = top_categories,
            seen_count       = int(len(np.unique(seen_products))) if len(seen_products) > 0 else 0,
            recommendations  = recommendations,
        )

    def _build_reason(
        self,
        category: str,
        cat_weight: float,
        price_weight: float,
        signal: str,
        score: float,
    ) -> str:
        """
        Compose a clear, specific explanation for each recommendation
        based on the actual GA-learned chromosome weights.
        """
        signal_labels = {
            "purchased": "previously purchased similar items",
            "rated":     "high ratings in similar products",
            "clicked":   "click history on similar products",
            "viewed":    "browsing history in this category",
            "unrated":   "general preference pattern",
        }
        signal_text = signal_labels.get(signal, "preference pattern")

        if cat_weight >= 0.65:
            strength = "very strong"
        elif cat_weight >= 0.40:
            strength = "strong"
        elif cat_weight >= 0.20:
            strength = "moderate"
        else:
            strength = "low"

        return (
            f"{strength.capitalize()} GA-learned affinity for {category} "
            f"(weight {cat_weight:.3f}), based on your {signal_text}. "
            f"Match score: {score * 100:.1f}%."
        )

    # ------------------------------------------------------------------
    # Helpers for the home page
    # ------------------------------------------------------------------

    def get_all_users(self) -> pd.DataFrame:
        return self.users.sort_values("user_id").copy()
