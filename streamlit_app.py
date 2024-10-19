import streamlit as st
import pandas as pd
import pickle

# Load your dataset
df = pd.read_csv('coimbatore_equalized_combined_restaurants.csv')

# Trim whitespace from column names
df.columns = df.columns.str.strip()

# Load the model
with open('cosine_similarity.pkl', 'rb') as file:
    cosine_sim = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf = pickle.load(file)

# Function to get recommendations based on user input
def get_recommendations(location, cuisine_type, food_type, price_range):
    filtered_df = df[
        (df['Location'] == location) &
        (df['Cuisine Type'] == cuisine_type) &
        (df['Type'] == food_type) &
        (df['Price (INR)'] >= price_range[0]) &
        (df['Price (INR)'] <= price_range[1])
    ]
    
    if filtered_df.empty:
        return "No recommendations found for the selected criteria."

    sample_size = min(5, len(filtered_df))
    return filtered_df[['Dish', 'Description', 'Price (INR)']].sample(sample_size)

# Styled Title and Description
st.markdown("<h1 style='text-align: center;'>Food Recommendation System 🍽️</h1>", unsafe_allow_html=True)

# User inputs
location = st.selectbox("Select Location", df['Location'].unique())
cuisine_type = st.selectbox("Select Cuisine Type", df['Cuisine Type'].unique())
food_type = st.selectbox("Select Type", df['Type'].unique())
price_range = st.slider("Select Price Range (INR)", min_value=0, max_value=int(df['Price (INR)'].max()), value=(200, 1000))

# Recommend food based on inputs
if st.button("Recommend Food"):
    recommendations = get_recommendations(location, cuisine_type, food_type, price_range)
    
    if isinstance(recommendations, str):
        st.write(recommendations)
    else:
        st.write("We recommend you try:")
        for index, row in recommendations.iterrows():
            # Use a distinct green background for better visibility
             st.success(
                f"**{row['Dish']}** - {row['Description']} (Price: ₹{row['Price (INR)']})"
            )
