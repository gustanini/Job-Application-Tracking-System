# data_loader.py
"""
This module handles loading and preprocessing of datasets.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_csv(file_path):
    """
    Load the CSV file containing job listings.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: DataFrame containing the job data.
    """
    try:
        data = pd.read_csv(file_path)
        print("CSV loaded successfully!")
        return data
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def load_cv(file_path):
    """
    Load the CV text file.
    Args:
        file_path (str): Path to the CV text file.
    Returns:
        str: The CV content as a string.
    """
    try:
        with open(file_path, 'r') as file:
            cv_content = file.read()
        print("CV loaded successfully!")
        return cv_content
    except Exception as e:
        print(f"Error loading CV: {e}")
        return None

def preprocess_csv(data):
    """
    Preprocess the job listings data.
    Args:
        data (pd.DataFrame): Raw job data.
    Returns:
        pd.DataFrame: Processed job data.
    """
    # Example: Drop rows with missing job descriptions
    data = data.dropna(subset=['Job Description'])
    data.reset_index(drop=True, inplace=True)
    print("CSV preprocessed successfully!")
    return data

def preprocess_cv(cv_content):
    """
    Preprocess the CV text.
    Args:
        cv_content (str): Raw CV content.
    Returns:
        str: Cleaned CV content.
    """
    # Example: Strip extra spaces or line breaks
    cv_content = cv_content.strip()
    print("CV preprocessed successfully!")
    return cv_content

def featurize_text(job_descriptions, cv_content):
    """
    Convert job descriptions and CV into numerical features using TF-IDF.
    Args:
        job_descriptions (pd.Series): List of job descriptions.
        cv_content (str): The user's CV content.
    Returns:
        tuple: TF-IDF matrix for job descriptions, and CV vector.
    """
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500) # converts text into numerical values based on term frequency and inverse document frequency
    job_features = vectorizer.fit_transform(job_descriptions)
    cv_vector = vectorizer.transform([cv_content])

    print("Text featurized successfully!")
    return job_features, cv_vector, vectorizer.get_feature_names_out()