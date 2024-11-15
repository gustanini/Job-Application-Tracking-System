# model.py
"""
This module handles keyword analysis and job-CV matching.
"""
from sklearn.metrics.pairwise import cosine_similarity

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
