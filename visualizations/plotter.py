# plotter.py
"""
This module handles plots and charts.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
import pandas as pd

def plot_keyword_frequency(feature_names, job_features, top_n=20):
    """
    Plot the top 20 keywords based on frequency.
    Args:
        feature_names (list): Feature names from TF-IDF.
        job_features (sparse matrix): TF-IDF matrix for job descriptions.
    """
    keyword_sums = np.array(job_features.sum(axis=0)).flatten()
    top_keywords = np.argsort(keyword_sums)[-top_n:]
    plt.figure(figsize=(10, 6))  # Increase figure size
    plt.barh([feature_names[i] for i in top_keywords], keyword_sums[top_keywords])
    plt.xlabel("Frequency")
    plt.title("Top 20 Keywords in Job Descriptions")
    plt.tight_layout()  # Adjust layout to prevent cropping
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
    plt.figure(figsize=(10, 6))  # Increase figure size
    plt.barh(labels, top_jobs['Similarity Score'])
    plt.xlabel("Similarity Score")
    plt.title(f"Top {top_n} Job Matches")
    plt.gca().invert_yaxis()  # Reverse order for better readability
    plt.tight_layout()  # Adjust layout to prevent cropping
    plt.show()

def plot_missing_keywords(missing_keywords, top_n=10):
    """
    Plot the top N missing keywords from the CV.
    Args:
        missing_keywords (list): Missing keywords with their scores.
        top_n (int): Number of top missing keywords to display.
    """
    keywords, scores = zip(*missing_keywords[:top_n])
    plt.figure(figsize=(10, 6))  # Increase figure size
    plt.barh(keywords, scores)
    plt.xlabel("Importance Score")
    plt.title(f"Top {top_n} Missing Keywords in CV")
    plt.gca().invert_yaxis()
    plt.tight_layout()  # Adjust layout to prevent cropping
    plt.show()

def plot_similarity_heatmap(ranked_jobs, top_n=10):
    """
    Create a heatmap showing similarity scores between the CV and job descriptions.
    Args:
        ranked_jobs (pd.DataFrame): Ranked job data.
        top_n (int): Number of top jobs to include in the heatmap.
    """
    top_jobs = ranked_jobs.head(top_n)
    plt.figure(figsize=(12, 6))  # Increase figure size
    sns.heatmap(
        top_jobs[['Similarity Score']].transpose(),
        annot=True,
        fmt=".2f",
        xticklabels=top_jobs['Job Title'],
        cmap="coolwarm"
    )
    plt.title(f"Similarity Scores for Top {top_n} Jobs")
    plt.xlabel("Job Titles")
    plt.ylabel("Similarity")
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
    plt.tight_layout()  # Adjust layout to prevent cropping
    plt.show()

def plot_interactive_keyword_comparison(feature_names, cv_vector, job_features, top_n=10):
    """
    Create an interactive plot comparing CV and job keywords.
    Args:
        feature_names (list): Feature names from TF-IDF.
        cv_vector (sparse matrix): TF-IDF vector for the CV.
        job_features (sparse matrix): TF-IDF matrix for job descriptions.
        top_n (int): Number of top keywords to display.
    """
    job_keywords = job_features.sum(axis=0).A1
    cv_keywords = cv_vector.toarray().flatten()

    # Combine into a DataFrame
    keyword_scores = [
        {"Keyword": feature_names[i], "Job Score": job_keywords[i], "CV Score": cv_keywords[i]}
        for i in range(len(feature_names))
    ]
    sorted_keywords = sorted(keyword_scores, key=lambda x: x['Job Score'], reverse=True)[:top_n]
    df = pd.DataFrame(sorted_keywords)

    # Create an interactive bar chart
    fig = px.bar(
        df,
        x="Keyword",
        y=["Job Score", "CV Score"],
        barmode="group",
        title="Keyword Comparison: CV vs. Job Descriptions",
        labels={"value": "TF-IDF Score", "Keyword": "Keyword"}
    )
    fig.show()
