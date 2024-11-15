from data import data_loader
from data.data_loader import load_csv, load_cv, preprocess_csv, preprocess_cv
from models.model import calculate_similarity, rank_jobs_by_similarity
from visualizations import plotter
from dashboard import dashboard
from utils import helpers
from data.data_loader import featurize_text
from visualizations.plotter import plot_keyword_frequency
from visualizations.plotter import plot_top_matches

# Paths to sample files
csv_path = "data/jobs.csv"
cv_path = "data/cv.txt"

# Load data
jobs_data = load_csv(csv_path)
#print(jobs_data.head())  # Display first few rows of the CSV

cv_data = load_cv(cv_path)
#print(cv_data)  # Display CV content

# Preprocess data
jobs_data = preprocess_csv(jobs_data)
#print(jobs_data.head())

cv_data = preprocess_cv(cv_data)
#print(cv_data)

# Featurize text
job_features, cv_vector, feature_names = featurize_text(jobs_data['Job Description'], cv_data)

# Visualize keyword frequency
plot_keyword_frequency(feature_names, job_features)

# Calculate similarity scores
similarity_scores = calculate_similarity(job_features, cv_vector)

# Rank jobs by similarity
ranked_jobs = rank_jobs_by_similarity(jobs_data, similarity_scores)
print(ranked_jobs[['Job Title', 'Similarity Score']].head())

# Visualize top matches
plot_top_matches(ranked_jobs)
