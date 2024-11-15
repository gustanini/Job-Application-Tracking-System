# data_loader.py
"""
This module handles loading and preprocessing of datasets.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
# import nltk

# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('wordnet')

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

    print("CSV preprocessed successfully!")
    return data

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

def remove_stop_words(text_series):
    """
    Remove stop words from a series of text data.
    Args:
        text_series (pd.Series): Series of text (job descriptions or CV).
    Returns:
        pd.Series: Text series with stop words removed.
    """
    # create custom list based off of results
    # user should iterate a couple of times to ensure only relevant words are present.
    CUSTOM_STOP_WORDS = {"experience", "required", "preferred", "responsibilities", "strong", "ability", "work", "skills", "team", "analytics", "requirements", "systems", "knowledge", "job", "years", "information", "quality", "company", "opportunity", "technical"}
    # append custom list
    ALL_STOP_WORDS = ENGLISH_STOP_WORDS.union(CUSTOM_STOP_WORDS)

    def clean_text(text):
        words = text.split()
        return " ".join([word for word in words if word.lower() not in ALL_STOP_WORDS])

    return text_series.apply(clean_text)

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