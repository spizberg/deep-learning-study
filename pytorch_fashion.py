
""" Code to see PyTorch behaviour compared to Tensorflow """

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as op
from sklearn.model_selection import train_test_split


def target_encoding(raw_target):
    """ One hot encode the target labels """

    rows = raw_target.shape[0]

    # converting class labels to 1-hot representations (targets)
    targets_1hot = np.zeros((raw_target.shape[0], np.max(raw_target) + 1))
    targets_1hot[np.arange(rows), raw_target] = 1

    return targets_1hot


# loading data set
_data = np.load("fashion_test_data.npy")
_targets = np.load("fashion_test_labels.npy")

train_set_data, val_set_data, train_set_targets, val_set_targets = train_test_split(_data, _targets,
                                                                                    train_size=0.75,
                                                                                    random_state=0,
                                                                                    stratify=_targets)


train_set_data = train_set_data.reshape((train_set_data.shape[0], 1, 28, 28)) / 255
val_set_data = val_set_data.reshape((val_set_data.shape[0], 1, 28, 28)) / 255


train_dataset = TensorDataset(torch.tensor(train_set_data, dtype=torch.float),
                              torch.tensor(train_set_targets, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(val_set_data, requires_grad=False, dtype=torch.float),
                            torch.as_tensor(val_set_targets))


train_data_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=4)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = op.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(50):

    running_loss = 0.0
    for i, data in enumerate(train_data_loader):

        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
        running_loss = 0.0

    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for data in val_data_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the %d test images: %d %%' % (total, 100 * correct / total))

print('Finished Training')

