import os.path as osp
import argparse
import numpy as np
import pandas as pd
import pickle
import tracemalloc
from sklearn.preprocessing import LabelEncoder


def get_dataset(dataset):
    if dataset == 'criteo':
        return CriteoDataset(dataset)
    elif dataset == 'avazu':
        return AvazuDataset(dataset)
    elif dataset == 'criteotb':
        return CriteoTBDataset(dataset)
    elif dataset == 'kdd12':
        return KDD12Dataset(dataset)


class CTRDataset(object):
    def __init__(self, path):
        self.path = path
        self.dtypes = {
            'dense': np.float32,
            'sparse': np.int32,
            'label': np.int32,
            'count': np.int32,
        }
        self.fpaths = {
            'dense': 'processed_dense.bin',
            'sparse': 'processed_sparse_sep.bin',
            'label': 'processed_label.bin',
            'count': 'processed_count.bin',
        }

    def join(self, fpath):
        return osp.join(self.path, fpath)

    def read_csv(self, fpath, **kwargs):
        path = self.join(fpath)
        assert osp.exists(path), f'Raw file path not exists: {path}'
        return pd.read_csv(path, **kwargs)

    def save_file(self, arr, key):
        arr = np.array(arr, dtype=self.dtypes[key])
        arr.tofile(self.join(self.fpaths[key]))

    def save_sparse(self, sparse):
        self.save_file(sparse, 'sparse')

    def save_dense(self, dense):
        self.save_file(dense, 'dense')

    def save_label(self, label):
        self.save_file(label, 'label')

    def save_count(self, count):
        self.save_file(count, 'count')

    def process_dense_feats(self, data, feats, inplace=True):
        if inplace:
            d = data
        else:
            d = data.copy()
        d = d[feats].fillna(0.0)
        for f in feats:
            d[f] = d[f].apply(lambda x: np.log(
                x+1) if x > 0 else 0)  # for criteo
        return d

    def process_sparse_feats(self, data, feats, inplace=True):
        if inplace:
            d = data
        else:
            d = data.copy()
        d = d[feats].fillna("0")
        for f in feats:
            label_encoder = LabelEncoder()
            d[f] = label_encoder.fit_transform(d[f])
        feature_cnt = 0
        counts = [d[f].nunique() for f in feats]
        return d, counts

    def process_data(self):
        raise NotImplementedError


class CriteoDataset(CTRDataset):
    def process_data(self):
        df = self.read_csv('train.txt', sep='\t', header=None)
        sparse_feats = df.columns[14:]
        dense_feats = df.columns[1:14]
        labels = df[0]
        dense = self.process_dense_feats(df, dense_feats)
        sparse, counts = self.process_sparse_feats(df, sparse_feats)
        self.save_dense(dense)
        self.save_sparse(sparse)
        self.save_label(labels)
        self.save_count(counts)


class CriteoTBDataset(CTRDataset):
    def process_data(self):
        ndays = 24
        num_sparse = 26
        for i in range(ndays):
            sparse = self.read_csv(f'day_{i}', header=None, sep='\t')
            labels = np.array(sparse[0], dtype=np.int32)
            labels.tofile(self.join(f'label_{i}.bin'))
            del labels
            dense = sparse[range(1, 14)]
            dense = dense.fillna(0.)
            dense = np.array(dense, dtype=np.float32)
            dense[dense < 0] = 0
            dense = np.log(dense + 1)
            dense.tofile(self.join(f'dense_{i}.bin'))
            del dense
            sparse = sparse[range(14, 40)]
            sparse = sparse.fillna('0')
            uniques = []
            for f in sparse:
                cur_set = set(sparse[f])
                uniques.append(list(cur_set))
                del cur_set
            with open(self.join(f'uniques_{i}.pkl'), 'wb') as fw:
                pickle.dump(uniques, fw)
            del uniques
            del sparse
        uniques = [[] for _ in range(num_sparse)]
        for i in range(ndays):
            with open(self.join(f'uniques_{i}.pkl'), 'rb') as fr:
                cur_uniques = pickle.load(fr)
            for j, uni in enumerate(cur_uniques):
                uniques[j] += uni
        for i in range(num_sparse):
            cur_set = set(uniques[i])
            if '0' in cur_set:
                cur_set.remove('0')
                cur_list = ['0'] + list(cur_set)
            else:
                cur_list = list(cur_set)
            del cur_set
            reverse_dict = {v: k for k, v in enumerate(cur_list)}
            del cur_list
            uniques[i] = reverse_dict
            del reverse_dict
        counts = [len(uni) for uni in uniques]
        npcounts = np.array(counts, dtype=np.int32)
        npcounts.tofile(self.join('processed_count.bin'))
        for i in range(ndays):
            sparse = self.read_csv(f'day_{i}', header=None, sep='\t')[
                range(14, 40)]
            sparse = sparse.fillna('0')
            for j, f in enumerate(range(14, 40)):
                sparse[f] = sparse[f].apply(lambda x: uniques[j][x])
            sparse = np.array(sparse, dtype=np.int32)
            sparse.tofile(self.join(f'sparse_{i}_sep.bin'))
            del sparse


class AvazuDataset(CTRDataset):
    def process_data(self):
        df = self.read_csv('train')
        sparse_feats = [
            col for col in df.columns if col not in ('click', 'id')]
        labels = df['click']
        sparse, counts = self.process_sparse_feats(df, sparse_feats)
        self.save_sparse(sparse)
        self.save_label(labels)
        self.save_count(counts)


class KDD12Dataset(CTRDataset):
    def process_data(self):
        df = self.read_csv('track2/training.txt', sep='\t', header=None)
        sparse_feats = [col for col in df.columns[1:]]
        labels = df[0].copy()
        sparse, counts = self.process_sparse_feats(df, sparse_feats)
        self.save_sparse(sparse)
        labels[labels > 1] = 1
        self.save_label(labels)
        self.save_count(counts)


if __name__ == '__main__':
    tracemalloc.start()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str, required=True,
        choices=['criteo', 'criteotb', 'avazu', 'kdd12']
    )
    args = parser.parse_args()
    dataset = get_dataset(args.dataset)
    dataset.process_data()
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 10**9} GB; Peak: {peak / 10**9} GB")
    tracemalloc.stop()
