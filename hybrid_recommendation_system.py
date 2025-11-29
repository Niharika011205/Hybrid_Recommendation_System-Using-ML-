"""
Amazon Prime Video - TRUE HYBRID Recommendation System
Content-Based Filtering + Collaborative Filtering
"""

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

# Load dataset
path = kagglehub.dataset_download("shivamb/amazon-prime-movies-and-tv-shows")
df = pd.read_csv(path + "/amazon_prime_titles.csv")

# Data Cleaning
df = df.drop_duplicates().reset_index(drop=True)
df['director'] = df['director'].fillna('Unknown Director')
df['cast'] = df['cast'].fillna('Unknown Cast')
df['listed_in'] = df['listed_in'].fillna('Unknown Genre')
df['description'] = df['description'].fillna('No description available')
df['release_year'] = df['release_year'].fillna(2000)

# ============================================================================
# PART 1: GENERATE REALISTIC USER RATINGS
# ============================================================================

# Create genre popularity scores (some genres are more popular)
genre_popularity = {
    'Action': 4.2, 'Comedy': 4.0, 'Drama': 3.8, 'Romance': 3.7,
    'Horror': 3.5, 'Documentary': 3.9, 'Animation': 4.3, 'Kids': 4.1,
    'Suspense': 4.0, 'Science Fiction': 4.2, 'Fantasy': 4.1
}

# Calculate base rating for each movie based on characteristics
def calculate_base_rating(row):
    """Generate realistic base rating based on movie features"""
    base = 3.5  # Average rating
    
    # Genre influence
    genres = row['listed_in'].split(',')
    genre_score = np.mean([genre_popularity.get(g.strip(), 3.5) for g in genres])
    
    # Recency bonus (newer movies tend to get higher ratings)
    year = row['release_year']
    if year >= 2018:
        recency_bonus = 0.3
    elif year >= 2015:
        recency_bonus = 0.2
    elif year >= 2010:
        recency_bonus = 0.1
    else:
        recency_bonus = 0
    
    # Type influence (TV shows slightly higher rated)
    type_bonus = 0.2 if row['type'] == 'TV Show' else 0
    
    # Calculate final base rating
    rating = (base * 0.4 + genre_score * 0.6 + recency_bonus + type_bonus)
    
    # Clip to valid range
    return np.clip(rating, 2.0, 5.0)

df['base_rating'] = df.apply(calculate_base_rating, axis=1)

# Generate user-item ratings matrix
np.random.seed(42)
n_users = 500  # Simulate 500 users
n_movies = len(df)

# Create sparse ratings (users don't rate all movies)
# Each user rates 5-50 movies randomly
ratings_data = []

for user_id in range(n_users):
    # Each user rates random number of movies
    n_ratings = np.random.randint(10, 50)
    movie_indices = np.random.choice(n_movies, n_ratings, replace=False)
    
    for movie_idx in movie_indices:
        base = df.iloc[movie_idx]['base_rating']
        # Add user preference noise (some users are harsh/generous)
        user_bias = np.random.normal(0, 0.3)
        # Add random noise
        noise = np.random.normal(0, 0.5)
        
        rating = base + user_bias + noise
        rating = np.clip(rating, 1.0, 5.0)
        rating = round(rating * 2) / 2  # Round to nearest 0.5
        
        ratings_data.append({
            'user_id': user_id,
            'movie_idx': movie_idx,
            'rating': rating
        })

ratings_df = pd.DataFrame(ratings_data)

# ============================================================================
# PART 2: CONTENT-BASED FILTERING
# ============================================================================

def clean_text(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = ' '.join(text.split())
    return text

df['description_clean'] = df['description'].apply(clean_text)

# TF-IDF for Descriptions
tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
tfidf_matrix = tfidf.fit_transform(df['description_clean'])
cosine_sim_tfidf = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Genres
df['genres_str'] = df['listed_in'].apply(lambda x: x.replace(',', ' '))
count_genres = CountVectorizer(stop_words='english')
genres_matrix = count_genres.fit_transform(df['genres_str'])
cosine_sim_genres = cosine_similarity(genres_matrix, genres_matrix)

# Cast
df['cast_str'] = df['cast'].apply(lambda x: x.replace(',', ' ') if x != 'Unknown Cast' else '')
count_cast = CountVectorizer(stop_words='english')
cast_matrix = count_cast.fit_transform(df['cast_str'])
cosine_sim_cast = cosine_similarity(cast_matrix, cast_matrix)

# Directors
df['director_str'] = df['director'].apply(lambda x: x.replace(',', ' ') if x != 'Unknown Director' else '')
count_director = CountVectorizer(stop_words='english')
director_matrix = count_director.fit_transform(df['director_str'])
cosine_sim_director = cosine_similarity(director_matrix, director_matrix)

# Combine content features
content_similarity = (
    0.40 * cosine_sim_tfidf +
    0.30 * cosine_sim_genres +
    0.20 * cosine_sim_cast +
    0.10 * cosine_sim_director
)

# ============================================================================
# PART 3: COLLABORATIVE FILTERING
# ============================================================================

# Create user-item matrix
user_item_matrix = ratings_df.pivot_table(
    index='user_id',
    columns='movie_idx',
    values='rating',
    fill_value=0
)

# Convert to sparse matrix for efficiency
user_item_sparse = csr_matrix(user_item_matrix.values)

# Use KNN for collaborative filtering (item-based)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
model_knn.fit(user_item_sparse.T)  # Transpose for item-based

# ============================================================================
# PART 4: HYBRID RECOMMENDATION FUNCTION
# ============================================================================

indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def get_collaborative_scores(movie_idx, n=50):
    """Get similar movies using collaborative filtering"""
    try:
        # Find similar movies based on user ratings
        distances, indices_knn = model_knn.kneighbors(
            user_item_sparse.T[movie_idx],
            n_neighbors=n+1
        )
        
        # Convert distances to similarity scores (1 - distance)
        similarities = 1 - distances.flatten()
        movie_indices = indices_knn.flatten()
        
        # Create score dictionary (exclude the movie itself)
        scores = {}
        for i, idx in enumerate(movie_indices[1:]):
            if idx < len(df):
                scores[idx] = similarities[i+1]
        
        return scores
    except:
        return {}

def hybrid_recommend(title, n=10, content_weight=0.5, collab_weight=0.5):
    """
    Hybrid recommendation combining content-based and collaborative filtering
    
    Parameters:
    -----------
    title : str
        Movie/show title
    n : int
        Number of recommendations
    content_weight : float
        Weight for content-based score (0-1)
    collab_weight : float
        Weight for collaborative filtering score (0-1)
    """
    try:
        idx = indices[title]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]
        
        # Get content-based scores
        content_scores = list(enumerate(content_similarity[idx]))
        content_dict = {i: score for i, score in content_scores}
        
        # Get collaborative filtering scores
        collab_dict = get_collaborative_scores(idx, n=50)
        
        # Combine scores
        hybrid_scores = {}
        all_indices = set(content_dict.keys()) | set(collab_dict.keys())
        
        for movie_idx in all_indices:
            if movie_idx == idx:  # Skip the movie itself
                continue
            
            content_score = content_dict.get(movie_idx, 0)
            collab_score = collab_dict.get(movie_idx, 0)
            
            # Weighted combination
            hybrid_score = (content_weight * content_score + 
                          collab_weight * collab_score)
            
            hybrid_scores[movie_idx] = hybrid_score
        
        # Sort by hybrid score
        sorted_scores = sorted(hybrid_scores.items(), 
                             key=lambda x: x[1], 
                             reverse=True)
        
        # Get top N
        top_indices = [i[0] for i in sorted_scores[:n]]
        
        return df.iloc[top_indices]['title'].tolist()
    
    except KeyError:
        return None

# ============================================================================
# PART 5: TEST THE SYSTEM
# ============================================================================
print("="*70)
print(" Testing Hybrid Recommendation System")
print("="*70)

test_movies = ["K.G.F: Chapter 1 (Telugu)", "The Grand Tour"]

for movie in test_movies:
    recommendations = hybrid_recommend(movie, n=10, content_weight=0.5, collab_weight=0.5)
    
    if recommendations:
        print(f"\nIf you liked: {movie}")
        print("You may also like:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print(f"\n'{movie}' not found in dataset")

print("\n" + "="*70)
