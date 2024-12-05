"""
Script that reads from raw Nowplaying-RS data and dumps into a pickle
file a heterogeneous graph with categorical and numeric features.
"""

import argparse
import os
import pickle

import pandas as pd
import scipy.sparse as ssp
from builder import PandasGraphBuilder
from data_utils import *

import dgl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str)
    parser.add_argument("out_directory", type=str)
    args = parser.parse_args()
    directory = args.directory
    out_directory = args.out_directory
    os.makedirs(out_directory, exist_ok=True)

    data = pd.read_csv(os.path.join(directory, "context_content_features.csv"))
    track_feature_cols = list(data.columns[1:13])
    data = data[
        ["user_id", "track_id", "created_at"] + track_feature_cols
    ].dropna()
    columns = list(data.columns)
    columns[1] = "item_id"
    columns[2] = "timestamp"
    print(data.columns)
    data.columns = columns
    print(data.columns)
    data = data.astype(
        {"user_id": "category", "item_id": "category", "artist_id": "category"}
    )
    data['user_id'], _ = pd.factorize(data['user_id'])
    data['item_id'], _ = pd.factorize(data['item_id'])
    data['artist_id'], _ = pd.factorize(data['artist_id'])
    data = data.astype(
        {"user_id": "category", "item_id": "category", "artist_id": "category"}
    )
    
    users = data[["user_id"]].drop_duplicates()
    tracks = data[["item_id"] + track_feature_cols].drop_duplicates()

    print(len(users))
    print(len(tracks))
    print(len(data))
    print(len(data) / len(users) / len(tracks))
    exit(0)

    assert tracks["item_id"].value_counts().max() == 1
    tracks = tracks.astype(
        {"mode": "int64", "key": "int64"}
    )
    events = data[["user_id", "item_id", "timestamp"]]
    events["timestamp"] = (
        events["timestamp"].values.astype("datetime64[s]").astype("int64")
    )

    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(users, "user_id", "user")
    graph_builder.add_entities(tracks, "item_id", "track")
    graph_builder.add_binary_relations(
        events, "user_id", "item_id", "listened"
    )
    graph_builder.add_binary_relations(
        events, "item_id", "user_id", "listened-by"
    )

    g = graph_builder.build()

    float_cols = []
    for col in tracks.columns:
        if col == "item_id":
            continue
        elif col == "artist_id":
            g.nodes["track"].data[col] = torch.LongTensor(
                tracks[col].cat.codes.values
            )
        elif tracks.dtypes[col] == "float64":
            float_cols.append(col)
        else:
            g.nodes["track"].data[col] = torch.LongTensor(tracks[col].values)
    g.nodes["track"].data["song_features"] = torch.FloatTensor(
        linear_normalize(tracks[float_cols].values)
    )
    g.edges["listened"].data["timestamp"] = torch.LongTensor(
        events["timestamp"].values
    )
    g.edges["listened-by"].data["timestamp"] = torch.LongTensor(
        events["timestamp"].values
    )

    n_edges = g.num_edges("listened")
    train_indices, val_indices, test_indices = train_test_split_by_time(
        events, "timestamp", "user_id"
    )
    train_g = build_train_graph(
        g, train_indices, "user", "track", "listened", "listened-by"
    )
    assert train_g.out_degrees(etype="listened").min() > 0
    val_matrix, test_matrix = build_val_test_matrix(
        g, val_indices, test_indices, "user", "track", "listened"
    )

    dgl.save_graphs(os.path.join(out_directory, "train_g.bin"), train_g)

    dataset = {
        "val-matrix": val_matrix,
        "test-matrix": test_matrix,
        "item-texts": {},
        "item-images": None,
        "user-type": "user",
        "item-type": "track",
        "user-to-item-type": "listened",
        "item-to-user-type": "listened-by",
        "timestamp-edge-column": "timestamp",
    }

    with open(os.path.join(out_directory, "data.pkl"), "wb") as f:
        pickle.dump(dataset, f)
