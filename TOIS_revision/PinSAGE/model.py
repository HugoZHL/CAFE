import argparse
import os
import pickle

import dgl

import evaluation
import layers
import numpy as np
import sampler as sampler_module
import torch
import torch.nn as nn
import torchtext
import tqdm
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, ntype, textsets, hidden_dims, n_layers, args):
        super().__init__()

        self.proj = layers.LinearProjector(
            full_graph, ntype, textsets, hidden_dims, args
        )
        self.sage = layers.SAGENet(hidden_dims, n_layers)
        self.scorer = layers.ItemToItemScorer(full_graph, ntype)

    def forward(self, pos_graph, neg_graph, blocks):
        h_item = self.get_repr(blocks)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_repr(self, blocks):
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        return h_item_dst + self.sage(blocks, h_item)

def get_class_from_model(model, class_path):
    attrs = class_path.split('.')
    
    current_obj = model
    for attr in attrs:
        current_obj = getattr(current_obj, attr)
    
    return current_obj

def eval(k_list, dataset, h_item, batch_size):
    for k in k_list:
        print(
            k, evaluation.evaluate_nn(dataset, h_item, k, batch_size)
        )

def train(dataset, args):
    g = dataset["train-graph"]
    val_matrix = dataset["val-matrix"].tocsr()
    test_matrix = dataset["test-matrix"].tocsr()
    item_texts = dataset["item-texts"]
    user_ntype = dataset["user-type"]
    item_ntype = dataset["item-type"]
    # item_ntype = "item"
    user_to_item_etype = dataset["user-to-item-type"]
    timestamp = dataset["timestamp-edge-column"]

    device = torch.device(args.device)

    # Assign user and movie IDs and use them as features (to learn an individual trainable
    # embedding for each entity)
    g.nodes[user_ntype].data["id"] = torch.arange(g.num_nodes(user_ntype))
    g.nodes[item_ntype].data["id"] = torch.arange(g.num_nodes(item_ntype))

    # Prepare torchtext dataset and Vocabulary
    textset = {}
    tokenizer = get_tokenizer(None)

    textlist = []
    batch_first = True

    for i in range(g.num_nodes(item_ntype)):
        for key in item_texts.keys():
            l = tokenizer(item_texts[key][i].lower())
            textlist.append(l)
    for key, field in item_texts.items():
        vocab2 = build_vocab_from_iterator(
            textlist, specials=["<unk>", "<pad>"]
        )
        textset[key] = (
            textlist,
            vocab2,
            vocab2.get_stoi()["<pad>"],
            batch_first,
        )

    # Sampler
    batch_sampler = sampler_module.ItemToItemBatchSampler(
        g, user_ntype, item_ntype, args.batch_size
    )
    neighbor_sampler = sampler_module.NeighborSampler(
        g,
        user_ntype,
        item_ntype,
        args.random_walk_length,
        args.random_walk_restart_prob,
        args.num_random_walks,
        args.num_neighbors,
        args.num_layers,
    )
    collator = sampler_module.PinSAGECollator(
        neighbor_sampler, g, item_ntype, textset
    )
    dataloader = DataLoader(
        batch_sampler,
        collate_fn=collator.collate_train,
        num_workers=args.num_workers,
    )
    dataloader_test = DataLoader(
        torch.arange(g.num_nodes(item_ntype)),
        batch_size=args.batch_size,
        collate_fn=collator.collate_test,
        num_workers=args.num_workers,
    )
    dataloader_it = iter(dataloader)

    # Model
    model = PinSAGEModel(
        g, item_ntype, textset, args.hidden_dims, args.num_layers, args
    ).to(device)
    # exit(0)
    
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Find the latest epoch_id from args.save_path
    existing_models = [
        f for f in os.listdir(args.save_path) if f.startswith("model_epoch_")
    ]
    if existing_models:
        latest_model = max(existing_models, key=lambda x: int(x.split("_")[2].split(".")[0]))
        latest_epoch_id = int(latest_model.split("_")[2].split(".")[0])
        checkpoint = torch.load(os.path.join(args.save_path, latest_model))
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        if args.compress_ratio > 1:
            model.proj.inputs.id.load_sketch(os.path.join(args.save_path, f"sketch_epoch_{(latest_epoch_id)}"))
        print(f"Resuming from epoch {latest_epoch_id}")
    else:
        latest_epoch_id = 0

    # print(len(dataloader_it))
    # exit(0)

    log_step = 1000
    save_step = 1
    eval_step = 25
    for epoch_id in range(latest_epoch_id, args.num_epochs):
        model.train()
        loss_accum = 0
        for batch_id in range(args.batches_per_epoch):
            pos_graph, neg_graph, blocks = next(dataloader_it)
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)

            loss = model(pos_graph, neg_graph, blocks).mean()
            opt.zero_grad()
            loss.backward()
            
            if args.compress_ratio > 1:
                model.proj.inputs.id.insert_all_grad()
            
            opt.step()
            
            loss_accum += loss.item()
            if (batch_id + 1) % log_step == 0:
                print(f"Epoch [{epoch_id + 1}/{args.num_epochs}], "
                      f"Step [{batch_id + 1}/{args.batches_per_epoch}], "
                      f"Loss: {loss.item():.4f}, "
                      f"Avg Loss: {loss_accum / log_step:.4f}, "
                      f"Remaining steps: {args.batches_per_epoch - batch_id - 1}", flush=True)
                loss_accum = 0

        # Save the model after each epoch
        if (epoch_id + 1) % save_step == 0:
            model_path = os.path.join(args.save_path, f"model_epoch_{(epoch_id + 1)}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
            }, model_path)
            if args.compress_ratio > 1:
                model.proj.inputs.id.save_sketch(os.path.join(args.save_path, f"sketch_epoch_{(epoch_id + 1)}"))

        if (epoch_id + 1) % eval_step == 0:
            # Evaluate
            model.eval()
            with torch.no_grad():
                item_batches = torch.arange(g.num_nodes(item_ntype)).split(
                    args.batch_size
                )
                h_item_batches = []
                for blocks in dataloader_test:
                    for i in range(len(blocks)):
                        blocks[i] = blocks[i].to(device)

                    h_item_batches.append(model.get_repr(blocks))
                h_item = torch.cat(h_item_batches, 0)

                print(
                    evaluation.evaluate_nn(dataset, h_item, args.k, args.batch_size), flush=True
                )
                eval([10, 20, 50, 100], dataset, h_item, args.batch_size)

    model.eval()
    with torch.no_grad():
        item_batches = torch.arange(g.num_nodes(item_ntype)).split(
            args.batch_size
        )
        h_item_batches = []
        for blocks in dataloader_test:
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)

            h_item_batches.append(model.get_repr(blocks))
        h_item = torch.cat(h_item_batches, 0)

        print(
            evaluation.evaluate_nn(dataset, h_item, args.k, args.batch_size)
        )
        eval([10, 20, 50, 100], dataset, h_item, args.batch_size)

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("--random-walk-length", type=int, default=2)
    parser.add_argument("--random-walk-restart-prob", type=float, default=0.5)
    parser.add_argument("--num-random-walks", type=int, default=10)
    parser.add_argument("--num-neighbors", type=int, default=3)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-dims", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device", type=str, default="cpu"
    )  # can also be "cuda:0"
    parser.add_argument("--num-epochs", type=int, default=60)
    parser.add_argument("--batches-per-epoch", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument('--compress-tech', choices=['cafe', 'hash'], default='cafe', type=str,
                        help='tech used for compressing')
    parser.add_argument('--compress-ratio', default=1, type=int,
                        help='compress ratio')
    args = parser.parse_args()

    # args.save_path = f"./results_yelp2020/model_{args.dataset_path}_rwlen{args.random_walk_length}_rwrprob{args.random_walk_restart_prob}_nrw{args.num_random_walks}_nn{args.num_neighbors}_nl{args.num_layers}_hd{args.hidden_dims}_bs{args.batch_size}_lr{args.lr}_ne{args.num_epochs}_bpe{args.batches_per_epoch}"
    if args.compress_tech == 'cafe':
        args.save_path = f"./results4/model_{args.dataset_path}_rwlen{args.random_walk_length}_rwrprob{args.random_walk_restart_prob}_nrw{args.num_random_walks}_nn{args.num_neighbors}_nl{args.num_layers}_hd{args.hidden_dims}_bs{args.batch_size}_lr{args.lr}_ne{args.num_epochs}_bpe{args.batches_per_epoch}"
    else:
        args.save_path = f"./results3/model_{args.dataset_path}_rwlen{args.random_walk_length}_rwrprob{args.random_walk_restart_prob}_nrw{args.num_random_walks}_nn{args.num_neighbors}_nl{args.num_layers}_hd{args.hidden_dims}_bs{args.batch_size}_lr{args.lr}_ne{args.num_epochs}_bpe{args.batches_per_epoch}"
    if args.compress_ratio > 1:
        args.save_path += f"_compress_{args.compress_tech}_{args.compress_ratio}"
    
    os.makedirs(args.save_path, exist_ok=True)

    # Load dataset
    data_info_path = os.path.join(args.dataset_path, "data.pkl")
    with open(data_info_path, "rb") as f:
        dataset = pickle.load(f)
    train_g_path = os.path.join(args.dataset_path, "train_g.bin")
    g_list, _ = dgl.load_graphs(train_g_path)
    dataset["train-graph"] = g_list[0]
    train(dataset, args)
