# Smart GA Store — Intelligent Algorithms Assignment

A Flask-based recommendation web app that applies a **Genetic Algorithm (GA)** inspired by the selected paper:

**A Genetic-Algorithms Matrix-Factorization Based Recommender System Model Using Tags (ETagMF)**.

This project uses the assignment dataset in `data/` and provides a modern, simple UI with clear GA result explanations.

---

## 1) Project Goal

The app simulates an e-commerce recommendation workflow:

- Read user/product/rating/behavior data from Excel files.
- Learn user preferences with a GA-based optimization process.
- Recommend top unseen products for each user.
- Explain recommendation quality and GA optimization behavior (RMSE evolution, chromosome weights).

---

## 2) Dataset Used

The system uses these required files from `data/`:

- `users.xlsx` → `user_id`, `age`, `country`
- `products.xlsx` → `product_id`, `category`, `price`
- `ratings.xlsx` → `user_id`, `product_id`, `rating`
- `behavior.xlsx` → `user_id`, `product_id`, `viewed`, `clicked`, `purchased`

---

## 3) Technical Stack

- **Backend:** Flask (Python)
- **Data processing:** Pandas, NumPy
- **Excel support:** OpenPyXL
- **Frontend:** Jinja templates + Tailwind CSS
- **Charts:** Chart.js

---

## 4) GA Method (Implemented Logic)

### Core idea

The app follows a simplified ETagMF-style approach:

- Build a product feature matrix (category one-hot + normalized price).
- Treat a chromosome as a user preference vector over features.
- Predict user-item score using dot product:
  - `predicted_score = user_weights · product_features`
- Optimize chromosome weights using GA to minimize **RMSE** against blended user targets.

### Target score for training

For each user-item interaction, target score combines:

- Explicit rating (normalized): weight `0.65`
- Implicit behavior (view/click/purchase): weight `0.35`

Behavior component:

- viewed → `0.15`
- clicked → `0.30`
- purchased → `0.55`

### Genetic operators

- **Initialization:** random genes in `[0, 1]`
- **Fitness:** RMSE (lower is better)
- **Selection:** Roulette wheel (inverse fitness probability)
- **Crossover:** Single-point crossover (`Pc`)
- **Mutation:** Per-gene random replacement (`Pm`)
- **Elitism:** top 20% kept each generation

---

## 5) GA Transparency in UI

The user recommendation page displays:

- GA parameters (`Np`, `Ni`, `Pc`, `Pm`)
- Initial RMSE vs Final RMSE
- RMSE improvement percentage
- Best generation index
- RMSE evolution curve across generations
- Chromosome feature weights (top categories + price sensitivity)
- Product-level explanation based on learned weights

This makes recommendation results readable and aligned with assignment expectations.

---

## 6) UI/UX Improvements Implemented

- Modern, clean layout with reusable glass cards.
- Improved header/navigation with active page highlighting.
- Dedicated team page with member cards and contribution details.
- Enhanced recommendation cards with rank, score badge, progress bar, and explanation text.
- Project switched to **Light Mode** with consistent color overrides.

---

## 7) Team Page and Work Distribution

The `/team` page includes all members with:

- Name
- Username
- Section
- Role
- Contribution to assignment implementation

This section documents how each student contributed to analysis, data preparation, GA implementation, UI/UX, integration/testing, and report writing.

---

## 8) Project Structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── users.xlsx
│   ├── products.xlsx
│   ├── ratings.xlsx
│   └── behavior.xlsx
├── app/
│   └── services/
│       └── recommender.py
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── user.html
│   └── team.html
└── static/
    └── styles.css
```

---

## 9) Run Instructions

From project root:

```bash
py -3 -m pip install -r requirements.txt
py -3 app.py
```

Open:

- `http://127.0.0.1:5000`

---

## 10) Notes and Scope

- This implementation is intentionally simple and readable for coursework.
- The GA design follows the paper conceptually while adapting to available assignment data.
- Recommendation quality is stochastic by nature, but tracked with RMSE and evolution history for clarity.

