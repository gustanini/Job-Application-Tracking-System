from flask import Flask, render_template, jsonify
from health_check.health_check import health_check
from logs.log_config import log_event

app = Flask(__name__, template_folder='templates')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/results")
def results():
    # Placeholder: Add logic to generate and display results
    data = {"Top Jobs": ["Job 1", "Job 2", "Job 3"], "Metrics": {"Precision": 0.85}}
    return render_template("results.html", data=data)

@app.route("/maintenance")
def maintenance():
    log_event("Starting health check...")
    checks = health_check()
    log_event(f"Health Check Results: {checks}")
    return render_template("maintenance.html", checks=checks)

if __name__ == "__main__":
    app.run(debug=True)
