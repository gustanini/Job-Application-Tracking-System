from data.data_loader import (
    load_csv, load_cv,
    preprocess_csv, preprocess_cv,
    featurize_text
)
from models.model import (
    calculate_similarity, rank_jobs_by_similarity,
    extract_top_keywords, find_missing_keywords,
    evaluate_similarity, evaluate_keyword_recommendations
)
from visualizations.plotter import (
    plot_keyword_frequency, plot_top_matches,
    plot_missing_keywords, plot_similarity_heatmap,
    plot_interactive_keyword_comparison, plot_evaluation_metrics
)
from logs.log_config import log_event
from config import CONFIG
from health_check.health_check import health_check
import traceback

def main():
    # Perform health check
    log_event("Starting health check...")
    checks = health_check()
    log_event(f"Health Check Results: {checks}")
    print("Health Check Results:", checks)

    # Exit if critical issues are found
    if not all(checks.values()):
        print("Critical issues detected during health check. Exiting.")
        log_event("Critical issues detected. Exiting application.", level="error")
        return

    # Log successful health check
    log_event("Health check passed. Proceeding with application.")

    try:
        # config variables
        csv_path = CONFIG["CSV_PATH"]
        cv_path = CONFIG["CV_PATH"]
        top_n_keywords = CONFIG["TOP_N_KEYWORDS"]
        top_n_jobs = CONFIG["TOP_N_JOBS"]

        log_event("Starting JATS application")

        # Load data
        jobs_data = load_csv(csv_path)
        cv_data = load_cv(cv_path)

        # Preprocess data
        jobs_data = preprocess_csv(jobs_data)
        cv_data = preprocess_cv(cv_data)

        # Featurize text
        job_features, cv_vector, feature_names = featurize_text(
            jobs_data['Job Description'], cv_data
        )

        # Visualize keyword frequency
        plot_keyword_frequency(feature_names, job_features)

        # Calculate similarity scores
        similarity_scores = calculate_similarity(job_features, cv_vector)

        # Rank jobs by similarity
        ranked_jobs = rank_jobs_by_similarity(jobs_data, similarity_scores)
        print(ranked_jobs[['Job Title', 'Similarity Score']].head())

        # Visualize top matches
        plot_top_matches(ranked_jobs)

        # Extract top keywords from job descriptions
        top_keywords = extract_top_keywords(feature_names, job_features, top_n=10)
        print("Top Keywords in Job Descriptions:", top_keywords)

        # Find missing keywords in the CV
        missing_keywords = find_missing_keywords(cv_vector, feature_names, job_features)
        print("Missing Keywords in CV:", missing_keywords[:10])  # Show top 10 missing keywords

        # Visualize missing keywords
        plot_missing_keywords(missing_keywords, top_n=10)

        # Visualize similarity heatmap
        plot_similarity_heatmap(ranked_jobs, top_n=5)

        # Visualize keyword comparison
        plot_interactive_keyword_comparison(
            feature_names, cv_vector,
            job_features, top_n=10
        )

        # Evaluate similarity scores
        similarity_metrics = evaluate_similarity(ranked_jobs, threshold=0.5)
        print("Similarity Metrics:", similarity_metrics)

        # Evaluate keyword recommendations
        recommendation_metrics = evaluate_keyword_recommendations(
            missing_keywords, top_n=10
        )
        print("Keyword Recommendation Metrics:", recommendation_metrics)

        # Combine metrics and visualize
        all_metrics = {**similarity_metrics, **recommendation_metrics}

        # Define the maximum values for scaling
        max_jobs = 2000  # Replace with actual maximum or dataset size

        # Normalize metrics
        normalized_metrics = {
            key: (value / max_jobs) * 100 if key == "Total Jobs" else value
            for key, value in all_metrics.items()
        }

        plot_evaluation_metrics(normalized_metrics)

    except Exception as e:
        # log errors
        log_event(f"Unexpected error: {e}", level="critical")
        print("An unexpected error occurred. Check the logs for details.")

if __name__ == "__main__":
    main()