import librosa
import numpy as np
import pickle as pkl
import os
import tqdm
import sys

import warnings
warnings.filterwarnings('ignore')


def getCQTSpec(path):
    raw_wave, sample_rate = librosa.load(path, sr=22050)
    win_len = 8192
    octaves = 8
    bins_per_semitone = 2
    bins_per_octave = 12 * bins_per_semitone
    data = np.abs(librosa.cqt(raw_wave, sr=sample_rate, hop_length=win_len // 2,
                              fmin=librosa.note_to_hz('C1'),
                              n_bins=bins_per_octave * octaves,
                              bins_per_octave=bins_per_octave))
    data = np.reshape(data, (data.shape[0], data.shape[1], 1))
    print(data.shape)
    return data.astype(np.float16)


def getKey():
    key_dict = {}
    DIR_path = "Datasets/giantsteps-key-dataset/annotations/key/"
    for filename in os.listdir(DIR_path):
        file_path = DIR_path + filename
        song_id, song_ext = os.path.splitext(filename)
        song_id = song_id.split(".")[0]
        if song_ext == ".key":
            with open(file_path, "r") as file:
                key_dict[song_id] = file.read().strip()
    with open("giantsteps_id_key.csv", "w") as file:
        for idx in sorted(key_dict):
            file.write(idx + "," + key_dict[idx] + "\n")


def getKeyFeatures():
    DIR_path = "Datasets/giantsteps-key-dataset/audio/"
    total_features = {}
    for filename in tqdm.tqdm(os.listdir(DIR_path)):
        file_path = DIR_path + filename
        song_id, song_ext = os.path.splitext(filename)
        song_id = song_id.split(".")[0]
        if song_ext == '.mp3' or song_ext == '.wav':
            song_feat = getCQTSpec(file_path)
            total_features[song_id] = song_feat
    with open(f'giantsteps_id_feat.pkl', 'wb') as f:
        pkl.dump(total_features, f)


if __name__ == "__main__":
    getKey()
    getKeyFeatures()

