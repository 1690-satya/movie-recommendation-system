import pandas as pd


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
