import torch
import pickle as pkl
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from models.shallowtempo import ShallowTempo
from models.deepsquare import DeepSquare

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model", help="name of model to use for training", type=str, choices=["deep_tempo", "shallow_tempo"])
parser.add_argument("epoch", help="number of epochs to train", type=int)
parser.add_argument("-l", "--learning_rate", help="learning rate for training", default=0.001, type=float)
parser.add_argument("-k", "--filter_size", help="size of the directional filter", default=12, type=int)
parser.add_argument("-p", "--drop_prob", help="dropout probability", default=0.25, type=float)
args = parser.parse_args()

"""# Dataset download and processing"""

def loadFromPKL(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data

fil_path = 'Processed_dataset/eball/'

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

"""# Training"""

epochs = args.epoch
lr = args.learning_rate
k =  args.filter_size
pD = args.drop_prob

if(args.model == "shallow_tempo"):
    model = ShallowTempo((1, 40, 256), 256, k, pD).to(device)  
elif(args.model == "deep_tempo"):
    model = DeepSquare((1, 40, 256), 256, k, pD).to(device)
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

"""# Saving the model"""

torch.save(model.state_dict(), f'trained_models/{args.model}_model.pt')

