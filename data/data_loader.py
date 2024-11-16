# data_loader.py
"""
This module handles loading and preprocessing of datasets.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import config
from logs.log_config import log_event, log_execution_time
from config import CONFIG

# import nltk

# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('wordnet')

@log_execution_time
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
        log_event(f"CSV loaded successfully from {file_path}")
        print(f"CSV loaded successfully from {file_path}")
        return data
    except Exception as e:
        log_event(f"Error loading CSV: {e}", level="error")
        return None

@log_execution_time
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
        log_event(f"CV loaded successfully from {file_path}")
        print(f"CV loaded successfully from {file_path}")
        return cv_content
    except Exception as e:
        print(f"Error loading CV: {e}")
        return None

@log_execution_time
def preprocess_csv(data):
    """
    Preprocess job listings data.
    Args:
        data (pd.DataFrame): Raw job data.
    Returns:
        pd.DataFrame: Processed job data.
    """
    # Drop rows with missing job descriptions
    data = data.dropna(subset=['Job Description'])
    data.reset_index(drop=True, inplace=True)

    # Remove stop words and lemmatize job descriptions
    data['Job Description'] = remove_stop_words(data['Job Description'])
    data['Job Description'] = lemmatize_text(data['Job Description'])

    log_event("CSV preprocessed successfully!")
    print("CSV preprocessed successfully!")
    return data

@log_execution_time
def preprocess_cv(cv_content):
    """
    Preprocess the CV text.
    Args:
        cv_content (str): Raw CV content.
    Returns:
        str: Processed CV content.
    """
    # Remove stop words and lemmatize CV content
    cv_content = remove_stop_words(pd.Series([cv_content])).iloc[0]
    cv_content = lemmatize_text(pd.Series([cv_content])).iloc[0]

    log_event("CV preprocessed successfully!")
    print("CV preprocessed successfully!")
    return cv_content

@log_execution_time
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

    log_event("Text featurized successfully!")
    print("Text featurized successfully!")
    return job_features, cv_vector, vectorizer.get_feature_names_out()

@log_execution_time
def remove_stop_words(text_series):
    """
    Remove stop words from a series of text data.
    Args:
        text_series (pd.Series): Series of text (job descriptions or CV).
    Returns:
        pd.Series: Text series with stop words removed.
    """
    # create custom list based off of results
    # user should iterate a couple of times to ensure only relevant words are present, make edits on config.py if necessary.
    CUSTOM_STOP_WORDS = config.CONFIG.get("CUSTOM_STOP_WORDS")
    # append custom list
    ALL_STOP_WORDS = ENGLISH_STOP_WORDS.union(CUSTOM_STOP_WORDS)

    def clean_text(text):
        words = text.split()
        return " ".join([word for word in words if word.lower() not in ALL_STOP_WORDS])

    return text_series.apply(clean_text)

@log_execution_time
def lemmatize_text(text_series):
    """
    Lemmatize words in a series of text data.
    Args:
        text_series (pd.Series): Series of text (job descriptions or CV).
    Returns:
        pd.Series: Text series with words lemmatized.
    """
    lemmatizer = WordNetLemmatizer()

    def lemmatize_words(text):
        words = word_tokenize(text)
        return " ".join([lemmatizer.lemmatize(word) for word in words])

    return text_series.apply(lemmatize_words)