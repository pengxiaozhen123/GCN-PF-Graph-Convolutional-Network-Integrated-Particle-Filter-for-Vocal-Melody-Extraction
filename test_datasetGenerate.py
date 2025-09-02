import numpy as np
import torch
import pandas as pd

np.set_printoptions(threshold=np.inf, precision=4)
torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None)


def load_features(feat_path, dtype=np.float32):
    feat_ori = pd.read_csv(feat_path, header=None)
    feat = np.array(feat_ori, dtype=dtype)
    return feat


def calculate_laplacian(matrix):
    matrix = matrix + torch.eye(matrix.size(0))
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    return normalized_laplacian


def GenerateDataset(data, mark, seq_len, time_len=None, normalize=True):
    if time_len is None:
        time_len = data.shape[0]
    if normalize:
        data = data
    val_size = time_len
    val_data = data[0:val_size]
    val_lable = mark[0:val_size]
    val_X, val_Y = list(), list()
    # val_X.append(np.array(val_data[0: 3].flatten()))
    # val_Y.append(np.array(val_lable[0]))
    for i in range(seq_len, len(val_data) - seq_len):
        val_X.append(np.array(val_data[i - seq_len: i + seq_len + 1].flatten()))
        val_Y.append(np.array(val_lable[i]))
    # val_X.append(np.array(val_data[-3:].flatten()))
    # val_Y.append(np.array(val_lable[-1]))
    val_Y = np.array(val_Y).reshape(-1, 63)
    return np.array(val_X), np.array(val_Y)


def GenerateTorchDataset(data, mark, time_len=None, seq_len=1, normalize=True):
    val_X, val_Y = GenerateDataset(data, mark, time_len=None, seq_len=2,
                                   normalize=True)
    val_X, val_Y = list(val_X), list(val_Y)
    val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(val_X),
                                                 torch.LongTensor(np.where(val_Y)[1]))
    return val_dataset
