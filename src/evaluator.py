class Evaluator:
    def precision_at_k(self, rec, rel, k):
        return len(set(rec[:k]) & set(rel)) / k
