# -*- coding: utf-8 -*-
"""MMT_Project_Tempo.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12KPwY0ETYON6-3RO-SfA9XuwfjRVyz0v
"""

import torch
import torch.nn as nn
import pickle as pkl
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

"""# Dataset download and processing"""

def loadFromPKL(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data

fil_path = 'Processed_dataset/'

xRaw = {}

for spkl in os.listdir(fil_path):
    pkl_id, pkl_ext = os.path.splitext(spkl)
    if(pkl_ext == '.pkl'):
        data = loadFromPKL(fil_path + spkl)
        for sid, feat in data.items():
            xRaw[int(sid)] = feat
    elif(pkl_ext == '.csv'):
        df = pd.read_csv(fil_path + spkl, header=None)

y_df_id, y_df_bpm = df[0].tolist(), df[1].tolist()
yRaw = {y_df_id[i]: y_df_bpm[i] for i in range(len(y_df_id))}

xTotal = []
yTotal = []

for sid in xRaw.keys():
    xTotal.append(xRaw[sid])
    yTotal.append(yRaw[sid])

class TempoDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __getitem__(self, key):
        st_off = np.random.randint(0, self.X[key].shape[1] - 255)
        return self.X[key][:, st_off: st_off + 256].astype(np.float32), self.y[key] - 30
    
    def __len__(self):
        return len(self.X)

xTrain, xVal, yTrain, yVal = train_test_split(xTotal, yTotal, test_size=0.2, random_state=42)
train_dataloader = DataLoader(TempoDataset(xTrain, yTrain), batch_size = 16, shuffle=True)
validation_dataloader = DataLoader(TempoDataset(xVal, yVal), batch_size = 16)

"""# Model"""

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Training the model on: {device}' + (f' ({torch.cuda.get_device_name(0)})' if torch.cuda.is_available() else ''))

class ShallowTempo(nn.Module):
    def __init__(self, input_size, output_size, k, p_dropout): # (1, F_t, T_t)
        super(ShallowTempo, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.k = k
        self.dropout = nn.Dropout(p_dropout)
        self.layer1 = nn.Conv2d(input_size[0], k, (1, 3), padding=1)

        self.poolingLayer = nn.AvgPool2d((input_size[1], 1))
        
        self.layer2 = nn.Conv2d(k, 64*k, (1, input_size[2]), padding=1)
        
        self.layer3 = nn.Conv2d(64*k, output_size, 1, padding=1) # (n, output_size, h', w')
        self.globalPooling = lambda x: x.mean([2, 3]) # (n, output_size, )
        self.relu = nn.ReLU()
    
    def forward(self, X): # X - (batch, C, H, W)
        output = self.relu(self.layer1(X)) # (n, k, h, w)
        output = self.dropout(output)

        output = self.poolingLayer(output) # (n, k, h', w')
        
        output = self.relu(self.layer2(output)) # (n, 64k, h', w')
        output = self.dropout(output)
        output = self.relu(self.layer3(output)) # (n, output_size, h', w')
        output = self.globalPooling(output) # (n, output_size)
        
        return output

"""# Training"""

epochs =  200#@param {type: "integer"}
lr = 0.001 #@param {type: "number"}
k =  12#@param {type: "integer"}
pD = 0.25 #@param {type: "number"}

model = ShallowTempo((1, 40, 256), 256, k, pD).to(device)
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

"""# Testing"""

tolerance = 0.04
model.eval()
correct_cases = 0
total_cases = 0
with torch.no_grad():
    for datap in validation_dataloader:
        datap[0] = torch.permute(datap[0], (0, 3, 1, 2))
        out = model(datap[0].to(device))
        oclass = torch.argmax(out, dim=1)
        for i in range(oclass.shape[0]):
            if(abs(oclass[i].item() - datap[1][i].item()) < tolerance * 256):
                correct_cases += 1
            total_cases += 1
print(f'Accuracy of model: {correct_cases / total_cases}')

"""# Saving the model"""

torch.save(model.state_dict(), 'model.pt')

"""# Demo"""

loaded_model = ShallowTempo((1, 40, 256), 256, k, pD).to(device)
loaded_model.load_state_dict(torch.load('model.pt', map_location=device))

def getTempo(x, model):
    x = x.astype(np.float32)
    x = torch.permute(torch.tensor(x), (2, 0, 1))
    x = torch.unsqueeze(x, dim=0)
    model.eval()
    with torch.no_grad():
        out = model(x.to(device))
        oclass = torch.argmax(out, dim=1)
    return oclass[0].item() + 30

ind = np.random.randint(0, len(xTotal))
print(ind, ':')
print(getTempo(xTotal[ind], model))
print(yTotal[ind])

