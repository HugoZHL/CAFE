import argparse
import pickle

import dgl

import numpy as np
import torch
from tqdm import trange


def prec(recommendations, ground_truth):
    n_users, n_items = ground_truth.shape
    K = recommendations.shape[1]
    user_idx = np.repeat(np.arange(n_users), K)
    item_idx = recommendations.flatten()
    relevance = ground_truth[user_idx, item_idx].reshape((n_users, K))
    hit = relevance.any(axis=1).mean()
    return hit

def ndcg(recommendations, ground_truth):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG)
    
    Args:
        recommendations: numpy array of shape (n_users, K) containing recommended item indices
        ground_truth: sparse matrix of shape (n_users, n_items) containing ground truth interactions
    
    Returns:
        float: average NDCG across all users
    """
    n_users, K = recommendations.shape
    
    # Get relevant items for each user from recommendations
    user_idx = np.repeat(np.arange(n_users), K)
    item_idx = recommendations.flatten()
    relevance = ground_truth[user_idx, item_idx].reshape((n_users, K))
    
    # Calculate DCG: rel_i / log2(i + 2)
    discounts = 1 / np.log2(np.arange(2, K + 2))
    discounts = np.tile(discounts, (n_users, 1))
    dcg = np.multiply(relevance, discounts).sum(axis=1)
    
    idcg = 1 / np.log2(2)
    
    res = np.divide(dcg, idcg)
    assert res.shape == dcg.shape
    
    return res.mean()

class LatestNNRecommender(object):
    def __init__(
        self, user_ntype, item_ntype, user_to_item_etype, timestamp, batch_size
    ):
        self.user_ntype = user_ntype
        self.item_ntype = item_ntype
        self.user_to_item_etype = user_to_item_etype
        self.batch_size = batch_size
        self.timestamp = timestamp

    def recommend(self, full_graph, K, h_user, h_item):
        """
        Return a (n_user, K) matrix of recommended items for each user
        """
        graph_slice = full_graph.edge_type_subgraph([self.user_to_item_etype])
        n_users = full_graph.num_nodes(self.user_ntype)
        latest_interactions = dgl.sampling.select_topk(
            graph_slice, 1, self.timestamp, edge_dir="out"
        )
        user, latest_items = latest_interactions.all_edges(
            form="uv", order="srcdst"
        )
        # each user should have at least one "latest" interaction
        assert torch.equal(user, torch.arange(n_users))

        recommended_batches = []
        user_batches = torch.arange(n_users).split(self.batch_size)
        for user_batch in user_batches:
            latest_item_batch = latest_items[user_batch].to(
                device=h_item.device
            )
            dist = h_item[latest_item_batch] @ h_item.t()
            # exclude items that are already interacted
            for i, u in enumerate(user_batch.tolist()):
                interacted_items = full_graph.successors(
                    u, etype=self.user_to_item_etype
                )
                dist[i, interacted_items] = -np.inf
            recommended_batches.append(dist.topk(K, 1)[1])

        recommendations = torch.cat(recommended_batches, 0)
        return recommendations


def evaluate_nn(dataset, h_item, k, batch_size):
    g = dataset["train-graph"]
    val_matrix = dataset["val-matrix"].tocsr()
    test_matrix = dataset["test-matrix"].tocsr()
    item_texts = dataset["item-texts"]
    user_ntype = dataset["user-type"]
    item_ntype = dataset["item-type"]
    user_to_item_etype = dataset["user-to-item-type"]
    timestamp = dataset["timestamp-edge-column"]

    rec_engine = LatestNNRecommender(
        user_ntype, item_ntype, user_to_item_etype, timestamp, batch_size
    )

    recommendations = rec_engine.recommend(g, k, None, h_item).cpu().numpy()
    return prec(recommendations, val_matrix), ndcg(recommendations, val_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("item_embedding_path", type=str)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    with open(args.dataset_path, "rb") as f:
        dataset = pickle.load(f)
    with open(args.item_embedding_path, "rb") as f:
        emb = torch.FloatTensor(pickle.load(f))
    print(evaluate_nn(dataset, emb, args.k, args.batch_size))