# =============================================
# ELITE Movie Recommendation System
# Includes: CF + Content + SVD + Hybrid Model
# =============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# =========================
# Data Loader
# =========================
class DataLoader:
    def __init__(self, movies_path, ratings_path):
        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)

    def merge(self):
        return pd.merge(self.ratings, self.movies, on='movieId')

# =========================
# Collaborative Filtering
# =========================
class CollaborativeFiltering:
    def __init__(self, data):
        self.data = data

    def build_matrix(self):
        self.matrix = self.data.pivot_table(index='userId', columns='title', values='rating').fillna(0)

    def compute_similarity(self):
        self.user_sim = cosine_similarity(self.matrix)
        self.item_sim = cosine_similarity(self.matrix.T)

    def user_based(self, user_id, k=5):
        sim = pd.Series(self.user_sim[user_id-1])
        top_users = sim.sort_values(ascending=False)[1:11]
        scores = np.dot(top_users.values, self.matrix.iloc[top_users.index])
        scores = pd.Series(scores, index=self.matrix.columns)
        return scores.sort_values(ascending=False).head(k)

    def item_based(self, user_id, k=5):
        user_ratings = self.matrix.iloc[user_id-1]
        scores = np.dot(self.item_sim, user_ratings)
        scores = pd.Series(scores, index=self.matrix.columns)
        return scores.sort_values(ascending=False).head(k)

# =========================
# Content-Based
# =========================
class ContentBased:
    def __init__(self, movies):
        self.movies = movies

    def build(self):
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.movies['genres'].fillna(''))
        self.sim = cosine_similarity(tfidf_matrix)

    def recommend(self, title, k=5):
        idx = self.movies[self.movies['title'] == title].index[0]
        scores = list(enumerate(self.sim[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:k+1]
        return self.movies['title'].iloc[[i[0] for i in scores]]

# =========================
#  SVD (Matrix Factorization)
# =========================
class SVDRecommender:
    def __init__(self, matrix):
        self.matrix = matrix

    def train(self, n_components=50):
        self.svd = TruncatedSVD(n_components=n_components)
        self.latent = self.svd.fit_transform(self.matrix)

    def recommend(self, user_id, k=5):
        user_vec = self.latent[user_id-1]
        scores = np.dot(self.latent, user_vec)
        scores = pd.Series(scores, index=self.matrix.index)
        return scores.sort_values(ascending=False).head(k)

# =========================
# Hybrid Recommender
# =========================
class HybridRecommender:
    def __init__(self, cf, cb, movies):
        self.cf = cf
        self.cb = cb
        self.movies = movies

    def recommend(self, user_id, base_movie, k=5):
        cf_rec = self.cf.item_based(user_id, k*2)
        cb_rec = self.cb.recommend(base_movie, k*2)

        combined = list(cf_rec.index) + list(cb_rec)
        return pd.Series(combined).value_counts().head(k)

# =========================
# Evaluation
# =========================
class Evaluator:
    def precision_at_k(self, rec, rel, k):
        return len(set(rec[:k]) & set(rel)) / k

# =========================
# Main
# =========================
if __name__ == "__main__":
    loader = DataLoader('data/movies.csv', 'data/ratings.csv')
    data = loader.merge()

    # CF
    cf = CollaborativeFiltering(data)
    cf.build_matrix()
    cf.compute_similarity()

    print("User CF:")
    print(cf.user_based(1))

    print("Item CF:")
    print(cf.item_based(1))

    # Content
    cb = ContentBased(loader.movies)
    cb.build()
    print("Content:")
    print(cb.recommend('Toy Story (1995)'))

    # SVD
    svd = SVDRecommender(cf.matrix)
    svd.train()
    print("SVD:")
    print(svd.recommend(1))

    # Hybrid
    hybrid = HybridRecommender(cf, cb, loader.movies)
    print("Hybrid:")
    print(hybrid.recommend(1, 'Toy Story (1995)'))

# =============================================
# ELITE FEATURES:
# - SVD Matrix Factorization
# - Hybrid Recommendation System
# - Multiple Models Comparison
# =============================================
