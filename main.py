import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train, X_test = X_train.reshape(60000,1,28,28), X_test.reshape(10000,1,28,28)
X_train, X_test = torch.from_numpy(X_train).float(), torch.from_numpy(X_test).float()
y_train, y_test = torch.from_numpy(y_train).long(), torch.from_numpy(y_test).long()
print(X_train.shape)

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1,28,kernel_size=3,padding=2)
        self.conv2 = nn.Conv2d(28,14,kernel_size=3,padding=2)
        self.conv3 = nn.Conv2d(14,10,kernel_size=3,padding=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(250, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, X):
        X = F.relu(F.max_pool2d(self.conv1(X), 2))
        X = F.relu(F.max_pool2d(self.conv2(X), 2))
        X = F.relu(F.max_pool2d(self.conv3(X), 2))
        X = self.flatten(X)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        return F.softmax(X, dim=1)



net = ConvNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


for epoch in range(20):

    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(loss)

print('Finished Training')