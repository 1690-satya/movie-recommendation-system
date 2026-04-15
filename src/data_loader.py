import os
import pandas as pd


class DataLoader:
    def __init__(self, movies_path, ratings_path):
        if not os.path.exists(movies_path):
            raise FileNotFoundError(f"Movies file not found: {movies_path}")
        if not os.path.exists(ratings_path):
            raise FileNotFoundError(f"Ratings file not found: {ratings_path}")
        
        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)

    def merge(self):
        return pd.merge(self.ratings, self.movies, on='movieId')
