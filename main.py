from data.data_loader import load_csv, load_cv, preprocess_csv, preprocess_cv, featurize_text
from models.model import calculate_similarity, rank_jobs_by_similarity, extract_top_keywords, find_missing_keywords
from visualizations.plotter import plot_keyword_frequency, plot_top_matches, plot_missing_keywords, plot_similarity_heatmap, plot_interactive_keyword_comparison
from visualizations import plotter
from dashboard import dashboard
from utils import helpers


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
plot_interactive_keyword_comparison(feature_names, cv_vector, job_features, top_n=10)