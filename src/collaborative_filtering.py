import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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
