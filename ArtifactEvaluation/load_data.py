import numpy as np
import os.path as osp
import torch
from bisect import bisect_right


class CTRDataset:
    def __init__(
        self,
        data_path,
        phase,
        max_ind_range=-1,
    ):
        self.sparse = self.read_sparse(data_path)
        self.dense = self.read_dense(data_path)
        self.label = self.read_label(data_path)
        self.counts = self.read_count(data_path)
        self.index = None
        self.get_split(phase)

    @property
    def num_sparse(self):
        raise NotImplementedError

    @property
    def num_dense(self):
        raise NotImplementedError

    @property
    def num_sample(self):
        raise NotImplementedError

    def check_path(self, path):
        if not osp.exists(path):
            raise AssertionError(f'Data not exists: {path}')

    def read_n_check(self, path, dtype, shape):
        self.check_path(path)
        result = np.memmap(path, dtype=dtype, mode='r').reshape(shape)
        return result

    def read_dense(self, data_path):
        if self.num_dense == 0:
            result = None
        else:
            dense_path = osp.join(data_path, 'processed_dense.bin')
            result = self.read_n_check(dense_path, np.float32, (self.num_sample, self.num_dense))
        return result

    def read_sparse(self, data_path):
        sparse_path = osp.join(data_path, 'processed_sparse_sep.bin')
        result = self.read_n_check(sparse_path, np.int32, (self.num_sample, self.num_sparse))
        return result

    def read_label(self, data_path):
        label_path = osp.join(data_path, 'processed_label.bin')
        result = self.read_n_check(label_path, np.int32, (self.num_sample,))
        return result

    def read_count(self, data_path):
        count_path = osp.join(data_path, 'processed_count.bin')
        self.check_path(count_path)
        result = np.fromfile(count_path, dtype=np.int32).reshape(self.num_sparse,)
        return result

    def get_split(self, phase):
        raise NotImplementedError

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                self[idx]
                for idx in range(
                    index.start or 0, index.stop or len(self), index.step or 1
                )
            ]
        if self.index is not None:
            index = self.index[index]
        data_c = np.array(self.sparse[index])
        dense_result = self.dense[index] if self.dense is not None else None
        return (data_c, dense_result, self.label[index])

    def __len__(self):
        return len(self.label)


class CriteoDataset(CTRDataset):
    @property
    def num_sparse(self):
        return 26

    @property
    def num_dense(self):
        return 13

    @property
    def num_sample(self):
        return 45840617

    def get_split(self, phase):
        train_len = self.num_sample * 6 // 7
        if phase == 'train':
            self.sparse = self.sparse[:train_len]
            self.dense = self.dense[:train_len]
            self.label = self.label[:train_len]
        elif phase == 'test':
            self.sparse = self.sparse[train_len:]
            self.dense = self.dense[train_len:]
            self.label = self.label[train_len:]


class CriteoTBDataset(CTRDataset):
    def __init__(
        self,
        data_path,
        phase,
        max_ind_range=-1,
    ):
        self.max_ind_range = max_ind_range
        self.phase = phase
        self.counts = self.read_count(data_path)
        self.index = 0
        if max_ind_range > 0:
            self.counts = np.minimum(self.counts, max_ind_range)

        train_n_day = 23
        if phase == 'train':
            self.sparse_file = []
            self.dense_file = []
            self.label_file = []
            self.num_entry = 0
            self.file_offset = np.zeros(train_n_day, dtype=np.int64)
            for day in range(train_n_day):
                c_mmap = self.read_n_check(
                    osp.join(data_path, f'sparse_{day}_sep.bin'),
                    np.int32, (-1, self.num_sparse)
                )
                d_mmap = self.read_n_check(
                    osp.join(data_path, f'dense_{day}.bin'),
                    np.float32, (-1, self.num_dense),
                )
                l_mmap = self.read_n_check(
                    osp.join(data_path, f'label_{day}.bin'),
                    np.int32, (-1,),
                )
                sz = l_mmap.shape[0]
                self.num_entry += sz
                self.file_offset[day] = self.num_entry
                self.sparse_file.append(c_mmap)
                self.dense_file.append(d_mmap)
                self.label_file.append(l_mmap)
        else:
            day = train_n_day
            self.sparse_file = self.read_n_check(
                osp.join(data_path, f'sparse_{day}_sep.bin'),
                np.int32, (-1, self.num_sparse)
            )
            self.dense_file = self.read_n_check(
                osp.join(data_path, f'dense_{day}.bin'),
                np.float32, (-1, self.num_dense),
            )
            self.label_file = self.read_n_check(
                osp.join(data_path, f'label_{day}.bin'),
                np.int32, (-1,),
            )
            self.num_entry = self.label_file.shape[0]

    def __len__(self):
        return self.num_entry

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                self[idx]
                for idx in range(
                    index.start or 0, index.stop or len(self), index.step or 1
                )
            ]
        if self.phase == 'train':
            file_idx = bisect_right(self.file_offset, index)
            sparse_file = self.sparse_file[file_idx]
            dense_file = self.dense_file[file_idx]
            label_file = self.label_file[file_idx]
            if file_idx > 0:
                index -= self.file_offset[file_idx - 1]
        if self.phase == 'test':
            sparse_file = self.sparse_file
            dense_file = self.dense_file
            label_file = self.label_file
        sparse_data = sparse_file[index]
        if self.max_ind_range != -1:
            sparse_data = sparse_data % self.max_ind_range
        return (sparse_data, dense_file[index], label_file[index])

    @property
    def num_sparse(self):
        return 26

    @property
    def num_dense(self):
        return 13

    @property
    def num_sample(self):
        return 4373472329


class AvazuDataset(CTRDataset):
    @property
    def num_sparse(self):
        return 22

    @property
    def num_dense(self):
        return 0

    @property
    def num_sample(self):
        return 40428967

    def get_split(self, phase):
        n_last_day = 4218938
        train_len = self.num_sample - n_last_day
        if phase == 'train':
            self.sparse = self.sparse[:train_len]
            self.label = self.label[:train_len]
        elif phase == 'test':
            self.sparse = self.sparse[train_len:]
            self.label = self.label[train_len:]


class KDD12Dataset(CTRDataset):
    @property
    def num_sparse(self):
        return 11

    @property
    def num_dense(self):
        return 0

    @property
    def num_sample(self):
        return 149639105

    def get_split(self, phase):
        np.random.seed(2023)
        tot_len = self.num_sample
        index = np.arange(tot_len)
        np.random.shuffle(index)

        test_size = int(0.1 * tot_len)
        if phase == 'train':
            split_index = index[test_size:]
        else:
            split_index = index[:test_size]
        self.index = split_index

    def __len__(self):
        return len(self.index)

def collate_wrapper(list_of_tuples):
    transposed_data = list(zip(*list_of_tuples))
    X_int = transposed_data[1]
    if len(X_int) > 0 and X_int[0] is not None:
        X_int = torch.tensor(np.array(X_int, dtype=np.float32), dtype=torch.float)
    else:
        X_int = None
    X_cat = torch.tensor(np.array(transposed_data[0]), dtype=torch.long)
    T = torch.tensor(transposed_data[2], dtype=torch.float32).view(-1, 1)

    batchSize = X_cat.shape[0]
    featureCnt = X_cat.shape[1]

    lS_i = [X_cat[:, i] for i in range(featureCnt)]
    lS_o = [torch.tensor(range(batchSize)) for _ in range(featureCnt)]

    return X_int, torch.stack(lS_o), torch.stack(lS_i), T


def make_datasets_and_loaders(args):

    dataset_cls = {
        'criteo': CriteoDataset,
        'criteotb': CriteoTBDataset,
        'avazu': AvazuDataset,
        'kdd12': KDD12Dataset,
    }[args.dataset]

    train_data = dataset_cls(
        args.data_path,
        'train',
        args.max_ind_range,
    )
    test_data = dataset_cls(
        args.data_path,
        'test',
        args.max_ind_range,
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.mini_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper,
        pin_memory=False,
        drop_last=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.test_mini_batch_size,
        shuffle=False,
        num_workers=args.test_num_workers,
        collate_fn=collate_wrapper,
        pin_memory=False,
        drop_last=False,
    )

    return train_data, train_loader, test_data, test_loader
