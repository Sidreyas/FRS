import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle

# Load your dataset
df = pd.read_csv('/home/maverick/Documents/PROJECTS/Food_Recom_Sys/coimbatore_equalized_combined_restaurants (1).csv')

# Create a TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Dish'])

# Calculate cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to save the model
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Save the cosine similarity matrix and the TF-IDF Vectorizer
save_model(cosine_sim, 'cosine_similarity.pkl')
save_model(tfidf, 'tfidf_vectorizer.pkl')

print("Model saved successfully!")
