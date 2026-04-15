# Movie Recommendation System

A comprehensive movie recommendation system implementing multiple collaborative filtering and content-based approaches.

## Features

- **Collaborative Filtering**: User-based and item-based recommendations using cosine similarity
- **Content-Based Filtering**: TF-IDF vectorization on movie genres
- **Matrix Factorization (SVD)**: Dimensionality reduction for latent factor recommendations
- **Hybrid Model**: Combined approach leveraging CF + Content-Based strengths

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and merging
│   ├── collaborative_filtering.py  # CF algorithms
│   ├── content_based.py        # Content-based filtering
│   ├── svd_recommender.py      # SVD matrix factorization
│   ├── hybrid_recommender.py   # Hybrid model
│   └── evaluator.py            # Evaluation metrics
├── data/                       # Dataset directory
├── main.py                     # Entry point
└── requirements.txt            # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

This project uses the [MovieLens dataset](https://grouplens.org/datasets/movielens/). Download and place in `data/`:
- `movies.csv`
- `ratings.csv`

## Usage

```bash
python main.py
```

## Algorithms

| Algorithm | Description |
|-----------|-------------|
| User-Based CF | Finds similar users and recommends what they liked |
| Item-Based CF | Recommends items similar to what the user rated highly |
| Content-Based | Recommends movies with similar genres |
| SVD | Matrix factorization for latent factor discovery |
| Hybrid | Combines item-based CF and content-based scores |

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
