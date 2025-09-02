import numpy as np
import torch
import pandas as pd

np.set_printoptions(threshold=np.inf, precision=4)
torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None)


def load_features(feat_path, dtype=np.float32):
    feat_ori = pd.read_csv(feat_path, header=None)
    feat = np.array(feat_ori, dtype=dtype)
    return feat


def load_adj(adj_path, dtype=np.float32):
    adj_ori = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_ori, dtype=dtype)
    return adj


def load_mark(mark_path, dtype=int):
    mark_ori = pd.read_csv(mark_path, header=None)
    mark = np.array(mark_ori, dtype=dtype)
    return mark


def calculate_laplacian(matrix):
    matrix = matrix  # - torch.eye(matrix.size(0))
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    return normalized_laplacian


def generate_dataset(
        data, mark, seq_len, time_len=None, split_ratio=0.99, normalize=True):
    if time_len is None:
        time_len = data.shape[0]
    if normalize:
        data = data
    train_size = int(time_len * split_ratio)
    train_data = data[0:train_size]
    train_mark = mark[0:train_size]
    test_data = data[train_size:time_len]
    test_mark = mark[train_size:time_len]
    train_X, train_Y, test_X, test_Y = list(), list(), list(), list()
    # train_X.append(np.array(train_data[0: 3].flatten()))
    # train_Y.append(np.array(train_mark[0]))
    for i in range(seq_len, len(train_data) - seq_len):
        train_X.append(np.array(train_data[i - seq_len: i + seq_len + 1].flatten()))
        train_Y.append(np.array(train_mark[i]))
    # train_X.append(np.array(train_data[-3:].flatten()))
    # train_Y.append(np.array(train_mark[-1]))
    # test_X.append(np.array(train_data[0: 3].flatten()))
    # test_Y.append(np.array(train_mark[0]))
    for i in range(seq_len, len(test_data) - seq_len):
        test_X.append(np.array(test_data[i - seq_len: i + seq_len + 1].flatten()))
        test_Y.append(np.array(test_mark[i]))
    # test_X.append(np.array(train_data[-3:].flatten()))
    # test_Y.append(np.array(train_mark[-1]))
    train_Y = np.array(train_Y).reshape(-1, 63)
    test_Y = np.array(test_Y).reshape(-1, 63)
    return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)


def generate_torch_datasets(data, mark, seq_len, time_len=None, split_ratio=0.99, normalize=True):
    train_X, train_Y, test_X, test_Y = generate_dataset(data, mark, seq_len, time_len=time_len,
                                                        split_ratio=split_ratio, normalize=normalize, )
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_X),
                                                   torch.LongTensor(np.where(train_Y)[1]))
    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(test_X),
                                                  torch.LongTensor(np.where(test_Y)[1]))
    return train_dataset, test_dataset
