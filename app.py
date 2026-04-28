from pathlib import Path

from flask import Flask, render_template, request

from app.services.recommender import GeneticRecommender


BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "data"

app        = Flask(__name__)
recommender = GeneticRecommender(data_dir=str(DATA_DIR))


@app.route("/")
def home():
    users = recommender.get_all_users()
    return render_template("index.html", users=users.to_dict(orient="records"))


@app.route("/user/<int:user_id>")
def user_recommendations(user_id: int):
    top_n = request.args.get("top_n", default=12, type=int)
    top_n = max(4, min(top_n, 30))

    # GA runs exactly once here — all view data comes from a single call
    page_data = recommender.get_user_page_data(user_id, top_n=top_n)
    return render_template("user.html", data=page_data, top_n=top_n)


@app.route("/team")
def team():
    team_members = [
        {
            "name": "نورالهدى فياض العلي",
            "username": "nour_alhda_239383",
            "section": "C3",
            "role": "Project Analysis Lead",
            "contribution": "Mapped assignment requirements to the selected paper and defined the GA-based recommendation approach used in implementation and reporting.",
        },
        {
            "name": "تسنيم فرحان خرفان",
            "username": "tasneem_218326",
            "section": "C2",
            "role": "Data Preparation Engineer",
            "contribution": "Prepared and validated users/products/ratings/behavior datasets, handled joins and quality checks, and documented preprocessing decisions.",
        },
        {
            "name": "جمعة مصطفى دويك",
            "username": "Jomah_240338",
            "section": "C3",
            "role": "GA Recommender Developer",
            "contribution": "Implemented the Genetic Algorithm core including fitness calculation, selection, crossover, mutation, and recommendation scoring logic.",
        },
        {
            "name": "غنى طبيخ",
            "username": "ghina_216066",
            "section": "C3",
            "role": "UI/UX Frontend Developer",
            "contribution": "Designed and improved user interfaces with Tailwind CSS, focusing on clear navigation, readable recommendation output, and modern visual consistency.",
        },
        {
            "name": "سارة حلبوني",
            "username": "sarah_251683",
            "section": "C3",
            "role": "Backend Integration & Testing",
            "contribution": "Integrated Flask routes with recommender services, verified end-to-end flow, and executed runtime checks to ensure stable behavior.",
        },
        {
            "name": "بيان عز الدين السبسبي",
            "username": "bayan_236201",
            "section": "C5",
            "role": "Documentation & Report Writer",
            "contribution": "Prepared the final technical report, explained GA-driven results, and structured references and findings according to assignment requirements.",
        },
    ]
    return render_template("team.html", team_members=team_members)


if __name__ == "__main__":
    app.run(debug=True)
