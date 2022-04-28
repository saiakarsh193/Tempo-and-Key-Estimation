import torch
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
import os
import pandas as pd
import numpy as np

from models.shallowtempo import ShallowTempo
from models.shallowkey import ShallowKey
from models.deepsquare import DeepSquare

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model", help="name of model to use for training", type=str, choices=["deep_tempo", "shallow_tempo", "shallow_tempo_control"])
parser.add_argument("-k", "--filter_size", help="size of the directional filter", default=12, type=int)
parser.add_argument("-p", "--drop_prob", help="dropout probability", default=0.25, type=float)
args = parser.parse_args()

"""# Dataset download and processing"""

def loadFromPKL(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data

fil_path = 'Processed_dataset/ball/'

xRaw = {}

for spkl in os.listdir(fil_path):
    pkl_id, pkl_ext = os.path.splitext(spkl)
    if(pkl_ext == '.pkl'):
        data = loadFromPKL(fil_path + spkl)
        for sid, feat in data.items():
            xRaw[sid] = feat
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

test_dataloader = DataLoader(TempoDataset(xTotal, yTotal), batch_size = 16, shuffle=True)

"""# Setting the parameters"""

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Testing the model on: {device}' + (f' ({torch.cuda.get_device_name(0)})' if torch.cuda.is_available() else ''))

k =  args.filter_size
pD = args.drop_prob


"""# Loading the model"""
if(args.model == "shallow_tempo"):
    loaded_model = ShallowTempo((1, 40, 256), 256, k, pD).to(device)  
elif(args.model == "deep_tempo"):
    loaded_model = DeepSquare((1, 40, 256), 256, k, pD).to(device)
elif(args.model == "shallow_tempo_control"):
    loaded_model = ShallowKey((1, 40, 256), 256, k, pD).to(device)
loaded_model.load_state_dict(torch.load(f'trained_models/{args.model}_model.pt', map_location=device))

"""# Testing"""

tolerance = 0.04
loaded_model.eval()
correct_cases = 0
total_cases = 0
with torch.no_grad():
    for datap in test_dataloader:
        datap[0] = torch.permute(datap[0], (0, 3, 1, 2))
        out = loaded_model(datap[0].to(device))
        oclass = torch.argmax(out, dim=1)
        for i in range(oclass.shape[0]):
            if(abs(oclass[i].item() - datap[1][i].item()) < tolerance * 256):
                correct_cases += 1
            total_cases += 1

print(f'Accuracy of {args.model} model on Ballroom dataset: {correct_cases / total_cases}')

