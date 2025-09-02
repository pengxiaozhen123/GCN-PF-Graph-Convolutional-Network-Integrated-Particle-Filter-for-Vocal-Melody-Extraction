from torch.utils.data import DataLoader
from test_datasetGenerate import *
from GCN_model import GCN
import torch

feat_path = r'G:\HPSS\medleydbtest.csv'
adj_path = r'D:\五帧\邻接矩阵\邻接矩阵5帧.csv'
mark_path = r'G:\台式机器数据\三个数据集融合训练特征\melodydbtestlabels.csv'
feat = load_features(feat_path)
adj = load_features(adj_path)
mark = load_features(mark_path)
val_dataset = GenerateTorchDataset(feat, mark, seq_len=2)
val_loader = DataLoader(val_dataset, batch_size=5000000, shuffle=False)
net = GCN(adj, 63)

criterion = torch.nn.CrossEntropyLoss()
# net.load_state_dict(torch.load(r"D:\多帧输入的邻接矩阵\不错位三帧.pth"))
net.load_state_dict(torch.load(r"D:\五帧\5帧模型.pth"))


#测试参数
total = sum([param.nelement() for param in net.parameters()])
print('Number of parameter:%.5fM' % (total/1e6))

net.eval()
truth = []
pre = []


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            features, labels = data
            outputs = net(features)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            truth.append(labels)
            pre.append(predicted)

    print('当前总数为:', total)
    print('Accuracy on test set: %.3f %%' % (100 * correct / total))
    print('*' * 20)


if __name__ == '__main__':
    test()
    truth = np.array(truth[0])
    pre = np.array(pre[0])
    print(len(pre))
#     # np.savetxt(r'D:\Mirex05-2022-2-28\混合真值.csv', truth.T, fmt='%.1f',
#     #             delimiter=',')
#     np.savetxt(r'G:\HPSS\结果\medleydb结果.csv', pre.T, fmt='%.0f', delimiter=',')

