import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tqdm
import os
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from shallowkey import ShallowKey


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)


def loadFromPKL(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data


data_path = "Processed_dataset/giantsteps/"
features = {}
features_file_path = data_path + "giantsteps_id_feat.pkl"
with open(features_file_path, "rb") as f:
    features = pkl.load(f)
labels_file_path = data_path + "giantsteps_id_key.csv"
df = pd.read_csv(labels_file_path, header=None)
label_map = {}
inverse_label_map = {}
y_id, y_label = df[0].to_list(), df[1].to_list()
labels = {}
label_id = 0
for i, label in enumerate(y_label):
    labels[y_id[i]] = label
    if label not in label_map:
        label_map[label] = label_id
        inverse_label_map[label_id] = label
        label_id += 1

xTotal = []
yTotal = []
for _id in features:
    xTotal.append(features[_id])
    yTotal.append(label_map[labels[int(_id)]])


class KeyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __getitem__(self, key):
        st_off = np.random.randint(0, self.X[key].shape[1] - 59)
        return self.X[key][:, st_off: st_off + 60].astype(np.float32), self.y[key]
    
    def __len__(self):
        return len(self.X)


xTrain, xVal, yTrain, yVal = train_test_split(xTotal, yTotal, test_size=0.2, random_state=42)
train_dataloader = DataLoader(KeyDataset(xTrain, yTrain), batch_size = 16, shuffle=True)
validation_dataloader = DataLoader(KeyDataset(xVal, yVal), batch_size = 16)

epochs =  200#@param {type: "integer"}
lr = 0.001 #@param {type: "number"}
k =  12#@param {type: "integer"}
pD = 0.5 #@param {type: "number"}

model = ShallowKey((1, 192, 60), 24, k, pD).to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_loss_values = []
valid_loss_values = []

for epoch in tqdm.tqdm(range(epochs)):
    # print("Epoch:", epoch + 1)
    total_train_loss = 0
    model.train()
    for datap in train_dataloader:
        datap[0] = torch.permute(datap[0], (0, 3, 1, 2))
        out = model(datap[0].to(device))
        loss = loss_function(out, datap[1].to(device))
        total_train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        total_valid_loss = 0
        for datap in validation_dataloader:
            datap[0] = torch.permute(datap[0], (0, 3, 1, 2))
            out = model(datap[0].to(device))
            loss = loss_function(out, datap[1].to(device))
            total_valid_loss += loss.item()
    # print(f"Total loss: {total_train_loss} \t Avg loss: {total_train_loss / len(train_dataloader)}")
    # print(f"Avg validation loss: {total_valid_loss / len(validation_dataloader)}")
    train_loss_values.append(total_train_loss / len(train_dataloader))
    valid_loss_values.append(total_valid_loss / len(validation_dataloader))

plt.plot(train_loss_values, c = 'blue', label = 'training loss')
plt.plot(valid_loss_values, c = 'red', label = 'validation loss')
plt.title('Training Progress')
plt.legend()
plt.show()

torch.save(model.state_dict(), 'Models/key_model.pt')
