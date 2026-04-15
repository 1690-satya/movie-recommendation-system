import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD


class SVDRecommender:
    def __init__(self, matrix):
        self.matrix = matrix

    def train(self, n_components=50):
        self.svd = TruncatedSVD(n_components=n_components)
        self.latent = self.svd.fit_transform(self.matrix)

    def recommend(self, user_id, k=5):
        user_vec = self.latent[user_id-1]
        scores = np.dot(self.latent, user_vec)
        scores = pd.Series(scores, index=self.matrix.columns)
        return scores.sort_values(ascending=False).head(k)
