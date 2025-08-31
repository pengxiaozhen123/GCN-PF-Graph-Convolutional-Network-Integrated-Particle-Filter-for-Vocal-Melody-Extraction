import torch
import torch.nn as nn
from tools_doc import calculate_laplacian
import numpy as np
import matplotlib.pyplot as plt

class GCN(nn.Module):
    def __init__(self, adj, n_class):
        super(GCN, self).__init__()
        self.register_buffer('laplacian', calculate_laplacian(torch.FloatTensor(adj)))
        self._num_nodes = adj.shape[0]
        self._n_class = n_class
        self.weights = nn.Parameter(torch.FloatTensor(self._num_nodes, self._n_class ))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('tanh'))

    def forward(self, inputs):
        support = torch.mm(inputs, self.laplacian)
        output = torch.spmm(support, self.weights)
        np.savetxt(r'D:\权重.csv', self.weights, fmt='%.3f',
                           delimiter=',')
        result = output.detach().numpy()
        # np.savetxt(r'D:\结果.csv', result, fmt='%.3f',
        #            delimiter=',')
        # plt.figure(figsize=(60, 60))
        # result = result.T
        # result = np.flipud(result)
        # # plt.imshow(result)
        # # plt.show()
        # # plt.savefig(r'D:\result.jpg')
        # np.savetxt(r'D:\20220314用后可删除\结果.csv', result, fmt='%.3f',
        #            delimiter=',')
        return output