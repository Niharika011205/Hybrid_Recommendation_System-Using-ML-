"""
Amazon Prime Video - Hybrid Recommendation System UI
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import re

import kagglehub

# ============================================================================
# LOAD AND PREPROCESS DATA
# ============================================================================

@st.cache_resource
def load_data():
    path = kagglehub.dataset_download("shivamb/amazon-prime-movies-and-tv-shows")
    df = pd.read_csv(path + "/amazon_prime_titles.csv")
    
    # Data Cleaning
    df = df.drop_duplicates().reset_index(drop=True)
    df['director'] = df['director'].fillna('Unknown Director')
    df['cast'] = df['cast'].fillna('Unknown Cast')
    df['listed_in'] = df['listed_in'].fillna('Unknown Genre')
    df['description'] = df['description'].fillna('No description available')
    df['release_year'] = df['release_year'].fillna(2000)
    
    return df

@st.cache_resource
def compute_recommendations(df):
    # Genre popularity scores
    genre_popularity = {
        'Action': 4.2, 'Comedy': 4.0, 'Drama': 3.8, 'Romance': 3.7,
        'Horror': 3.5, 'Documentary': 3.9, 'Animation': 4.3, 'Kids': 4.1,
        'Suspense': 4.0, 'Science Fiction': 4.2, 'Fantasy': 4.1
    }
    
    # Calculate base rating
    def calculate_base_rating(row):
        base = 3.5
        genres = row['listed_in'].split(',')
        genre_score = np.mean([genre_popularity.get(g.strip(), 3.5) for g in genres])
        year = row['release_year']
        if year >= 2018:
            recency_bonus = 0.3
        elif year >= 2015:
            recency_bonus = 0.2
        elif year >= 2010:
            recency_bonus = 0.1
        else:
            recency_bonus = 0
        type_bonus = 0.2 if row['type'] == 'TV Show' else 0
        rating = (base * 0.4 + genre_score * 0.6 + recency_bonus + type_bonus)
        return np.clip(rating, 2.0, 5.0)
    
    df['base_rating'] = df.apply(calculate_base_rating, axis=1)
    
    # Generate user ratings
    np.random.seed(42)
    n_users = 500
    n_movies = len(df)
    ratings_data = []
    
    for user_id in range(n_users):
        n_ratings = np.random.randint(10, 50)
        movie_indices = np.random.choice(n_movies, n_ratings, replace=False)
        
        for movie_idx in movie_indices:
            base = df.iloc[movie_idx]['base_rating']
            user_bias = np.random.normal(0, 0.3)
            noise = np.random.normal(0, 0.5)
            rating = base + user_bias + noise
            rating = np.clip(rating, 1.0, 5.0)
            rating = round(rating * 2) / 2
            ratings_data.append({
                'user_id': user_id,
                'movie_idx': movie_idx,
                'rating': rating
            })
    
    ratings_df = pd.DataFrame(ratings_data)
    
    # Content-Based Filtering
    def clean_text(text):
        if pd.isna(text):
            return ''
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    df['description_clean'] = df['description'].apply(clean_text)
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
    tfidf_matrix = tfidf.fit_transform(df['description_clean'])
    cosine_sim_tfidf = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    df['genres_str'] = df['listed_in'].apply(lambda x: x.replace(',', ' '))
    count_genres = CountVectorizer(stop_words='english')
    genres_matrix = count_genres.fit_transform(df['genres_str'])
    cosine_sim_genres = cosine_similarity(genres_matrix, genres_matrix)
    
    df['cast_str'] = df['cast'].apply(lambda x: x.replace(',', ' ') if x != 'Unknown Cast' else '')
    count_cast = CountVectorizer(stop_words='english')
    cast_matrix = count_cast.fit_transform(df['cast_str'])
    cosine_sim_cast = cosine_similarity(cast_matrix, cast_matrix)
    
    df['director_str'] = df['director'].apply(lambda x: x.replace(',', ' ') if x != 'Unknown Director' else '')
    count_director = CountVectorizer(stop_words='english')
    director_matrix = count_director.fit_transform(df['director_str'])
    cosine_sim_director = cosine_similarity(director_matrix, director_matrix)
    
    content_similarity = (
        0.40 * cosine_sim_tfidf +
        0.30 * cosine_sim_genres +
        0.20 * cosine_sim_cast +
        0.10 * cosine_sim_director
    )
    
    # Collaborative Filtering
    user_item_matrix = ratings_df.pivot_table(
        index='user_id',
        columns='movie_idx',
        values='rating',
        fill_value=0
    )
    
    user_item_sparse = csr_matrix(user_item_matrix.values)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
    model_knn.fit(user_item_sparse.T)
    
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    return content_similarity, model_knn, user_item_sparse, indices, df

# ============================================================================
# RECOMMENDATION FUNCTIONS
# ============================================================================

def get_collaborative_scores(movie_idx, model_knn, user_item_sparse, n=50):
    try:
        distances, indices_knn = model_knn.kneighbors(
            user_item_sparse.T[movie_idx],
            n_neighbors=n+1
        )
        similarities = 1 - distances.flatten()
        movie_indices = indices_knn.flatten()
        scores = {}
        for i, idx in enumerate(movie_indices[1:]):
            if idx < len(df):
                scores[idx] = similarities[i+1]
        return scores
    except:
        return {}

def hybrid_recommend(title, content_similarity, model_knn, user_item_sparse, indices, df, n=10, content_weight=0.5, collab_weight=0.5):
    try:
        idx = indices[title]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]
        
        content_scores = list(enumerate(content_similarity[idx]))
        content_dict = {i: score for i, score in content_scores}
        collab_dict = get_collaborative_scores(idx, model_knn, user_item_sparse, n=50)
        
        hybrid_scores = {}
        all_indices = set(content_dict.keys()) | set(collab_dict.keys())
        
        for movie_idx in all_indices:
            if movie_idx == idx:
                continue
            content_score = content_dict.get(movie_idx, 0)
            collab_score = collab_dict.get(movie_idx, 0)
            hybrid_score = (content_weight * content_score + collab_weight * collab_score)
            hybrid_scores[movie_idx] = hybrid_score
        
        sorted_scores = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = [i[0] for i in sorted_scores[:n]]
        
        return df.iloc[top_indices][['title', 'type', 'release_year', 'listed_in']].reset_index(drop=True)
    
    except KeyError:
        return None

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(
    page_title="Amazon Prime Video Recommendations",
    page_icon="🎬",
    layout="wide"
)

# Load data
st.title("🎬 Amazon Prime Video - Hybrid Recommendations")
st.markdown("Get personalized movie and TV show recommendations using a hybrid approach!")

# Initialize session state
if 'data_loaded' not in st.session_state:
    with st.spinner("Loading and preprocessing data... This may take a moment."):
        df = load_data()
        content_similarity, model_knn, user_item_sparse, indices, df = compute_recommendations(df)
        st.session_state.data_loaded = True
        st.session_state.content_similarity = content_similarity
        st.session_state.model_knn = model_knn
        st.session_state.user_item_sparse = user_item_sparse
        st.session_state.indices = indices
        st.session_state.df = df
else:
    content_similarity = st.session_state.content_similarity
    model_knn = st.session_state.model_knn
    user_item_sparse = st.session_state.user_item_sparse
    indices = st.session_state.indices
    df = st.session_state.df

# Sidebar
st.sidebar.header("⚙️ Settings")
content_weight = st.sidebar.slider("Content-Based Weight", 0.0, 1.0, 0.5, 0.1)
collab_weight = st.sidebar.slider("Collaborative Filtering Weight", 0.0, 1.0, 0.5, 0.1)
n_recommendations = st.sidebar.slider("Number of Recommendations", 5, 20, 10, 1)

# Search
st.header("🔍 Find Recommendations")
search_term = st.text_input("Enter a movie or TV show you liked:", placeholder="e.g., K.G.F: Chapter 1 (Telugu)")

if search_term:
    # Find matching titles
    matching_titles = df[df['title'].str.contains(search_term, case=False, na=False)]['title'].unique()
    
    if len(matching_titles) > 0:
        # If exact match or close match, show recommendations
        if search_term in df['title'].values:
            selected_title = search_term
        else:
            # Show dropdown for selection
            selected_title = st.selectbox("Select a title:", matching_titles)
        
        if st.button("Get Recommendations", type="primary"):
            with st.spinner("Generating recommendations..."):
                recommendations = hybrid_recommend(
                    selected_title,
                    content_similarity,
                    model_knn,
                    user_item_sparse,
                    indices,
                    df,
                    n=n_recommendations,
                    content_weight=content_weight,
                    collab_weight=collab_weight
                )
            
            if recommendations is not None:
                st.success(f"Found {len(recommendations)} recommendations for '{selected_title}'!")
                
                for idx, row in recommendations.iterrows():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"**{row['title']}**")
                    with col2:
                        st.write(f"_{row['type']}_")
                    with col3:
                        st.write(f"⭐ {row['release_year']}")
                    st.markdown(f"*{row['listed_in']}*")
                    st.divider()
            else:
                st.error("Movie/TV show not found in dataset.")
    else:
        st.warning("No matching titles found. Try a different name.")

# Show some popular titles
st.header("🎬 Popular Titles in Dataset")
st.markdown("Looking for inspiration? Here are some popular titles:")
popular_titles = df['title'].sample(min(10, len(df))).tolist()
cols = st.columns(5)
for i, title in enumerate(popular_titles):
    with cols[i % 5]:
        st.write(title)
