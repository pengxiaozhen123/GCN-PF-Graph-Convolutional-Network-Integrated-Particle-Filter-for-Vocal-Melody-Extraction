from torch.utils.data import DataLoader
import torch.optim as optim
from tools_doc import *
from GCN_model import GCN

#feat_path = r'D:\20220612\特征.csv'
# adj_path = r'D:\12binsperoctave\邻接矩阵_3帧.csv'
#mark_path = r'D:\20220612\标签.csv'
feat_path = r'G:\HPSS\训练集.csv'
adj_path = r'D:\五帧\邻接矩阵\邻接矩阵5帧.csv'
mark_path = r'G:\HPSS\hpss标签.csv'
feat = load_features(feat_path)
adj = load_adj(adj_path)
mark = load_mark(mark_path)
train_dataset, test_dataset = generate_torch_datasets(feat, mark, seq_len=2)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, shuffle=False)
model = GCN(adj, 63)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
truth = []
pre = []
accuracy = []
train_loss = []


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()
        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 256 == 255:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 255))
            train_loss.append(running_loss)
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        truth.clear()
        pre.clear()
        for data in test_loader:
            features, labels = data
            outputs = model(features)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            truth.append(labels)
            pre.append(predicted)
    print('当前总数为:', total)
    print('Accuracy on test set: %.3f %%' % (100 * correct / total))
    print('*' * 20)


if __name__ == '__main__':
    for epoch in range(50):
        train(epoch)
        test()
        if epoch == 49:
            truth = np.array(truth)
            # np.savetxt(r'D:\三个数据集融合训练特征\只有melodydb做训练集\真值3帧.csv', truth.T, fmt='%.1f',
            #            delimiter=',')
            # pre = np.array(pre)
            # np.savetxt(r'D:\三个数据集融合训练特征\只有melodydb做训练集\预测3帧.csv', pre.T, fmt='%.0f',
            #            delimiter=',')
            # np.savetxt(r'D:\三个数据集融合训练特征\只有melodydb做训练集\loss3帧.csv', train_loss, fmt='%.3f',
            #            delimiter=',')
            torch.save(model.state_dict(), r"G:\HPSS\HPSS模型.pth")
