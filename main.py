import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data_train = pd.read_csv("/content/Digit Recognizer  - train.csv")
data_train.isnull().sum()
data_train = data_train.dropna()
print(data_train.isnull().sum())
x_train = data_train.iloc[:, 1:].values.reshape(-1, 28, 28, 1).astype("float32")
y_train = data_train.label.values
features_train, features_val, targets_train, targets_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)
featuresVal = torch.from_numpy(features_val)
targetsVal = torch.from_numpy(targets_val).type(torch.LongTensor)

train_dataset = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
val_dataset = torch.utils.data.TensorDataset(featuresVal, targetsVal)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)

class AutoEncoder(nn.Module):
  def __init__(self):
    super(AutoEncoder, self).__init__()
    self.encoder = nn.Sequential(
        nn.Linear(28 * 28, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
    )

    self.decoder = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 28 * 28),
    )
  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = AutoEncoder().cuda()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)
criterion = nn.MSELoss()

num_epochs = 30
val_loss_list = []
for epoch in range(num_epochs):
    autoencoder.train()
    val_loss = 0
    total = 0

    for i, (imgs, _) in enumerate(train_loader):
        x = imgs.view(-1, 784)
        y = imgs.view(-1, 784)
        pred = autoencoder(x.cuda())
        loss = criterion(pred, y.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    autoencoder.eval()
    with torch.no_grad():
        for i, (imgs, _) in enumerate(val_loader):
            x = imgs.view(-1, 784)
            y = imgs.view(-1, 784)
            val_pred = autoencoder(x.cuda())
            loss = criterion(val_pred, y.cuda())
            val_loss += loss.item()
            total += len(imgs)
            val_loss_list.append(val_loss / total)
        print(f"epoch:{epoch + 1}, loss:{val_loss / total}\n")

plt.plot(val_loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("AutoEncoder - val: Loss vs Number of iteration")
plt.show()

data_test = pd.read_csv("/content/Digit Recognizer - test.csv")
print(data_test.isnull().sum())
data_test = data_test.dropna()
x_test = torch.from_numpy(data_test.astype("float32").values)

autoencoder.eval()
test_loss_data = 0
test_total = 0
test_loss_list = []
with torch.no_grad():
    for i, data in enumerate(x_test):
        x = data.view(-1, 784)
        y = data.view(-1, 784)
        test_pred = autoencoder(x.cuda())
        test_loss += criterion(test_pred, y.cuda())

        test_total += len(data)
        test_loss_list.append(test_loss.cpu() / test_total)

plt.plot(test_loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("AutoEncoder - test: Loss vs Number of iteration")
plt.xticks([])
plt.yticks([])
plt.show()

img = x_test[0].reshape(28, 28)
plt.figure(figsize=(4, 4))
plt.imshow(img)
plt.show()





