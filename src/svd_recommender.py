import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class SVDRecommender:
    def __init__(self, matrix):
        self.matrix = matrix

    def train(self, n_components=50, n_clusters=5):
        self.svd = TruncatedSVD(n_components=n_components)
        self.latent = self.svd.fit_transform(self.matrix)
        
        # K-Means clustering on latent user factors
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.user_clusters = self.kmeans.fit_predict(self.latent)
        
        # Calculate Silhouette Score
        self.silhouette = silhouette_score(self.latent, self.user_clusters)
        print(f"Silhouette Score: {self.silhouette:.3f}")

    def recommend(self, user_id, k=5):
        user_cluster = self.user_clusters[user_id-1]
        # Get all users in the same cluster
        cluster_users = np.where(self.user_clusters == user_cluster)[0]
        # Average ratings of movies by users in the cluster
        cluster_ratings = self.matrix.iloc[cluster_users].mean(axis=0)
        return cluster_ratings.sort_values(ascending=False).head(k)
