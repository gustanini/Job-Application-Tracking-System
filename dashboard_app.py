import os
from flask import Flask, render_template, send_from_directory
import matplotlib.pyplot as plt

app = Flask(__name__)

# Path for saving static images
STATIC_DIR = "dashboard/static"
os.makedirs(STATIC_DIR, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/results")
def results():
    # Placeholder data for results
    top_jobs = ["Job 1", "Job 2", "Job 3"]
    similarity_scores = [0.8, 0.75, 0.7]

    # Generate and save static plot
    img_path = os.path.join(STATIC_DIR, "top_jobs.png")
    plt.figure(figsize=(8, 6))
    plt.bar(top_jobs, similarity_scores, color="skyblue")
    plt.title("Top Job Matches")
    plt.ylabel("Similarity Score")
    plt.xlabel("Jobs")
    plt.tight_layout()
    plt.savefig(img_path)  # Save plot as static image
    plt.close()

    return render_template("results.html", image="top_jobs.png", jobs=top_jobs)

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
