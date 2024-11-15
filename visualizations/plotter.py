# plotter.py
"""
This module handles plots and charts.
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_keyword_frequency(feature_names, job_features):
    """
    Plot the top 20 keywords based on frequency.
    Args:
        feature_names (list): Feature names from TF-IDF.
        job_features (sparse matrix): TF-IDF matrix for job descriptions.
    """
    keyword_sums = np.array(job_features.sum(axis=0)).flatten()
    top_keywords = np.argsort(keyword_sums)[-20:]
    plt.barh([feature_names[i] for i in top_keywords], keyword_sums[top_keywords])
    plt.xlabel("Frequency")
    plt.title("Top 20 Keywords in Job Descriptions")
    plt.show()

def plot_top_matches(ranked_jobs, top_n=5):
    """
    Plot the top N job matches based on similarity score.
    Args:
        ranked_jobs (pd.DataFrame): Ranked job data.
        top_n (int): Number of top matches to display.
    """
    top_jobs = ranked_jobs.head(top_n)
    # Create unique labels for jobs
    labels = [
        f"{title} ({score:.2f})"
        for title, score in zip(top_jobs['Job Title'], top_jobs['Similarity Score'])
    ]
    plt.barh(labels, top_jobs['Similarity Score'])
    plt.xlabel("Similarity Score")
    plt.title(f"Top {top_n} Job Matches")
    plt.gca().invert_yaxis()  # Reverse order for better readability
    plt.show()

def plot_missing_keywords(missing_keywords, top_n=10):
    """
    Plot the top N missing keywords from the CV.
    Args:
        missing_keywords (list): Missing keywords with their scores.
        top_n (int): Number of top missing keywords to display.
    """
    keywords, scores = zip(*missing_keywords[:top_n])
    plt.barh(keywords, scores)
    plt.xlabel("Importance Score")
    plt.title(f"Top {top_n} Missing Keywords in CV")
    plt.gca().invert_yaxis()
    plt.show()
