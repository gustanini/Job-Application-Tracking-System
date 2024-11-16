import os
from flask import Flask, render_template, send_from_directory
import matplotlib
import matplotlib.pyplot as plt

# Force Matplotlib to use the non-interactive Agg backend
matplotlib.use("Agg")

app = Flask(__name__)

# Path for saving static images
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/results")
def results():
    # Placeholder data
    top_jobs = ["Job 1", "Job 2", "Job 3"]
    similarity_scores = [0.8, 0.75, 0.7]
    keywords = ["Keyword 1", "Keyword 2", "Keyword 3", "Keyword 4"]
    keyword_scores = [10, 8, 7, 6]

    # Generate static images
    plots = []

    # 1. Keyword Frequency
    keyword_img_path = os.path.join(STATIC_DIR, "keyword_frequency.png")
    plt.figure(figsize=(8, 6))
    plt.barh(keywords, keyword_scores, color="skyblue")
    plt.title("Keyword Frequency")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(keyword_img_path)
    plt.close()
    plots.append("keyword_frequency.png")

    # 2. Missing Keywords
    missing_img_path = os.path.join(STATIC_DIR, "missing_keywords.png")
    plt.figure(figsize=(8, 6))
    plt.barh(keywords[::-1], keyword_scores[::-1], color="orange")
    plt.title("Missing Keywords")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(missing_img_path)
    plt.close()
    plots.append("missing_keywords.png")

    # 3. Similarity Heatmap
    heatmap_img_path = os.path.join(STATIC_DIR, "similarity_heatmap.png")
    plt.figure(figsize=(8, 6))
    plt.imshow([[0.8, 0.75, 0.7]], cmap="coolwarm", aspect="auto")
    plt.colorbar(label="Similarity Score")
    plt.xticks(range(len(top_jobs)), top_jobs, rotation=45, ha="right")
    plt.yticks([])
    plt.title("Similarity Heatmap")
    plt.tight_layout()
    plt.savefig(heatmap_img_path)
    plt.close()
    plots.append("similarity_heatmap.png")

    # 4. Top Matches
    matches_img_path = os.path.join(STATIC_DIR, "top_matches.png")
    plt.figure(figsize=(8, 6))
    plt.bar(top_jobs, similarity_scores, color="green")
    plt.title("Top Job Matches")
    plt.ylabel("Similarity Score")
    plt.tight_layout()
    plt.savefig(matches_img_path)
    plt.close()
    plots.append("top_matches.png")

    # 5. Keyword Comparison
    comparison_img_path = os.path.join(STATIC_DIR, "keyword_comparison.png")
    plt.figure(figsize=(8, 6))
    plt.bar(keywords, keyword_scores, color="purple", label="Job")
    plt.bar(keywords, [5, 4, 3, 2], color="pink", label="CV", alpha=0.7)
    plt.title("Keyword Comparison: CV vs Job")
    plt.ylabel("TF-IDF Score")
    plt.xlabel("Keywords")
    plt.legend()
    plt.tight_layout()
    plt.savefig(comparison_img_path)
    plt.close()
    plots.append("keyword_comparison.png")

    return render_template("results.html", plots=plots)

@app.route("/maintenance")
def maintenance():
    # Placeholder data for maintenance
    checks = {"CSV File Loaded": True, "CV File Loaded": True, "Logging Active": True}
    return render_template("maintenance.html", checks=checks)

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True)
