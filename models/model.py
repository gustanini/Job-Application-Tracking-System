# model.py
"""
This module handles keyword analysis and job-CV matching.
"""
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from logs.log_config import log_execution_time

@log_execution_time
def calculate_similarity(job_features, cv_vector):
    """
    Calculate cosine similarity between job descriptions and the CV.
    Args:
        job_features (sparse matrix): TF-IDF matrix for job descriptions.
        cv_vector (sparse matrix): TF-IDF vector for the CV.
    Returns:
        list: List of similarity scores for each job description.
    """
    similarities = cosine_similarity(job_features, cv_vector)
    return similarities.flatten()

@log_execution_time
def rank_jobs_by_similarity(jobs_data, similarity_scores):
    """
    Rank job descriptions based on similarity to the CV.
    Args:
        jobs_data (pd.DataFrame): Job descriptions and metadata.
        similarity_scores (list): Similarity scores for each job description.
    Returns:
        pd.DataFrame: Jobs ranked by similarity score.
    """
    jobs_data['Similarity Score'] = similarity_scores
    ranked_jobs = jobs_data.sort_values(by='Similarity Score', ascending=False)
    return ranked_jobs

@log_execution_time
def extract_top_keywords(feature_names, job_features, top_n=10):
    """
    Extract top keywords from job descriptions based on TF-IDF scores.
    Args:
        feature_names (list): Feature names from TF-IDF.
        job_features (sparse matrix): TF-IDF matrix for job descriptions.
        top_n (int): Number of top keywords to extract.
    Returns:
        list: Top keywords and their scores.
    """
    keyword_sums = np.array(job_features.sum(axis=0)).flatten()
    top_indices = np.argsort(keyword_sums)[-top_n:]
    top_keywords = [(feature_names[i], keyword_sums[i]) for i in reversed(top_indices)]
    return top_keywords

@log_execution_time
def find_missing_keywords(cv_vector, feature_names, job_features):
    """
    Identify keywords present in job descriptions but missing in the CV.
    Args:
        cv_vector (sparse matrix): TF-IDF vector for the CV.
        feature_names (list): Feature names from TF-IDF.
        job_features (sparse matrix): TF-IDF matrix for job descriptions.
    Returns:
        list: Missing keywords with their importance scores.
    """
    cv_keywords = cv_vector.toarray().flatten()
    job_keywords = job_features.sum(axis=0).A1

    missing_keywords = [
        (feature_names[i], job_keywords[i])
        for i in range(len(feature_names))
        if cv_keywords[i] == 0 and job_keywords[i] > 0
    ]
    return sorted(missing_keywords, key=lambda x: x[1], reverse=True)


@log_execution_time
def evaluate_similarity(ranked_jobs, threshold=0.5):
    """
    Evaluate the similarity scores to assess accuracy.
    Args:
        ranked_jobs (pd.DataFrame): Ranked job data.
        threshold (float): Minimum similarity score for a job to be considered relevant.
    Returns:
        dict: Evaluation metrics (precision, recall, etc.).
    """
    relevant_jobs = ranked_jobs[ranked_jobs['Similarity Score'] >= threshold]
    total_relevant = len(relevant_jobs)
    total_jobs = len(ranked_jobs)

    # Example metrics
    precision = total_relevant / total_jobs if total_jobs > 0 else 0
    metrics = {
        "Total Jobs": total_jobs,
        "Relevant Jobs": total_relevant,
        "Precision": precision,
    }
    return metrics

@log_execution_time
def evaluate_keyword_recommendations(missing_keywords, top_n=10):
    """
    Evaluate the quality of missing keyword recommendations.
    Args:
        missing_keywords (list): Missing keywords with their importance scores.
        top_n (int): Number of recommendations to consider.
    Returns:
        dict: Evaluation metrics for keyword recommendations.
    """
    top_recommendations = missing_keywords[:top_n]
    avg_importance_score = sum(score for _, score in top_recommendations) / len(
        top_recommendations) if top_recommendations else 0

    metrics = {
        "Top Recommendations": len(top_recommendations),
        "Average Importance Score": avg_importance_score,
    }
    return metrics
