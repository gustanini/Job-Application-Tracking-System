# model.py
"""
This module handles keyword analysis and job-CV matching.
"""
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
