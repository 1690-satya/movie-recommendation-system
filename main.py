from src.data_loader import DataLoader
from src.collaborative_filtering import CollaborativeFiltering
from src.content_based import ContentBased
from src.svd_recommender import SVDRecommender
from src.hybrid_recommender import HybridRecommender


def main():
    loader = DataLoader('data/movies.csv', 'data/ratings.csv')
    data = loader.merge()

    # CF
    cf = CollaborativeFiltering(data)
    cf.build_matrix()
    cf.compute_similarity()

    print("User CF:")
    print(cf.user_based(1))

    print("\nItem CF:")
    print(cf.item_based(1))

    # Content
    cb = ContentBased(loader.movies)
    cb.build()
    print("\nContent:")
    print(cb.recommend('Toy Story (1995)'))

    # SVD
    svd = SVDRecommender(cf.matrix)
    svd.train()
    print("\nSVD:")
    print(svd.recommend(1))

    # Hybrid
    hybrid = HybridRecommender(cf, cb, loader.movies)
    print("\nHybrid:")
    print(hybrid.recommend(1, 'Toy Story (1995)'))


if __name__ == "__main__":
    main()
