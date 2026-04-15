import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


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
