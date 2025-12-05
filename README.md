# Amazon Prime Video - Hybrid Recommendation System

A sophisticated recommendation engine that combines **Content-Based Filtering** and **Collaborative Filtering** to suggest movies and TV shows from Amazon Prime Video's catalog.

## Overview

This project implements a true hybrid recommendation system that leverages both content similarity (genres, cast, directors, descriptions) and user behavior patterns (collaborative filtering) to provide personalized recommendations.

## Features

- **Content-Based Filtering**: Analyzes movie metadata including:
  - Descriptions (TF-IDF vectorization)
  - Genres
  - Cast members
  - Directors

- **Collaborative Filtering**: Uses K-Nearest Neighbors (KNN) algorithm with user-item ratings matrix to find similar movies based on user preferences

- **Hybrid Approach**: Combines both methods with configurable weights for optimal recommendations

- **Realistic User Ratings**: Generates synthetic user ratings based on:
  - Genre popularity
  - Release year (recency bias)
  - Content type (Movie vs TV Show)
  - User-specific preferences and biases

## Dataset

Uses the [Amazon Prime Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/amazon-prime-movies-and-tv-shows) dataset from Kaggle, loaded via `kagglehub`.

## Installation

```bash
pip install pandas numpy scikit-learn scipy kagglehub
```

## Usage

```python
# Get recommendations for a movie/show
recommendations = hybrid_recommend(
    title="K.G.F: Chapter 1 (Telugu)",
    n=10,
    content_weight=0.5,
    collab_weight=0.5
)
```

### Parameters

- `title`: Movie or TV show title
- `n`: Number of recommendations to return (default: 10)
- `content_weight`: Weight for content-based filtering (0-1, default: 0.5)
- `collab_weight`: Weight for collaborative filtering (0-1, default: 0.5)

## How It Works

1. **Data Preprocessing**: Cleans and prepares the dataset, handling missing values
2. **User Rating Generation**: Creates a realistic user-item ratings matrix with 500 simulated users
3. **Content Similarity**: Calculates cosine similarity across multiple content features
4. **Collaborative Filtering**: Uses KNN to find similar items based on user rating patterns
5. **Hybrid Scoring**: Combines both approaches with weighted averaging

## Visualizations

The project includes several data visualizations:

- `movies_vs_tvshows.png` - Distribution of content types
- `release_years.png` - Content release timeline
- `top_10_genres.png` - Most popular genres
- `directors_actors.png` - Top directors and actors
- `wordcloud.png` - Word cloud of descriptions

## Example Output

```
If you liked: K.G.F: Chapter 1 (Telugu)
You may also like:
  1. [Similar Movie 1]
  2. [Similar Movie 2]
  ...
```

## Technical Details

- **Similarity Metrics**: Cosine similarity for both content and collaborative features
- **Sparse Matrix**: Uses `scipy.sparse.csr_matrix` for efficient memory usage
- **Algorithm**: K-Nearest Neighbors with cosine metric for collaborative filtering
- **Feature Weighting**: 
  - Description: 40%
  - Genres: 30%
  - Cast: 20%
  - Directors: 10%

## License

This project is for educational purposes.
