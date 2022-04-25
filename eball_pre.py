import xml.etree.ElementTree as ET
import librosa
import numpy as np
import pickle as pkl
import os
import tqdm
import sys

import warnings
warnings.filterwarnings('ignore')

def getBPM():
    bpm_dict = {}
    DIR_PATH = 'Datasets/extendedballroom_v1.1/'
    tree_root = ET.parse(DIR_PATH + 'extendedballroom_v1.1.xml').getroot()
    for genre_node in tree_root:
        for song_node in genre_node:
            song_id = song_node.get('id')
            song_bpm = song_node.get('bpm')
            bpm_dict[song_id] = song_bpm
    with open('eball_id_bpm.csv', 'w') as f:
        for id, bpm in bpm_dict.items():
            f.write(str(id) + ',' + str(bpm) + '\n')

def getMelSpec(path):
    raw_wave, sample_rate = librosa.load(path, sr=11025)
    win_len = 1024
    data = librosa.feature.melspectrogram(y=raw_wave, sr=sample_rate, 
                                          n_fft=win_len, hop_length=(win_len // 2), 
                                          power=1, n_mels=40, fmin=20, fmax=5000)
    data = np.reshape(data, (data.shape[0], data.shape[1], 1))
    return data.astype(np.float16)

def getTempoFeatures():
    DIR_PATH = 'Datasets/extendedballroom_v1.1/'
    total_features = {}
    for p1 in os.listdir(DIR_PATH):
        p1_path = DIR_PATH + p1 + '/'
        if(os.path.isdir(p1_path)):
            print(f'Extracting temporal features for {p1} in {DIR_PATH}')
            for song in tqdm.tqdm(os.listdir(p1_path)):
                song_path = p1_path + song
                song_id, song_ext = os.path.splitext(song)
                if(song_ext == '.mp3'):
                    song_feat = getMelSpec(song_path)
                    total_features[song_id] = song_feat
    with open('eball_id_feat.pkl', 'wb') as f:
        pkl.dump(total_features, f)

def getTempoFeatures_Indv(sub_dir):
    sub_dir = sub_dir.capitalize()
    DIR_PATH = 'Datasets/extendedballroom_v1.1/' + sub_dir + '/'
    total_features = {}
    print(f'Extracting temporal features in {DIR_PATH}')
    for song in tqdm.tqdm(os.listdir(DIR_PATH)):
        song_path = DIR_PATH + song
        song_id, song_ext = os.path.splitext(song)
        if(song_ext == '.mp3'):
            song_feat = getMelSpec(song_path)
            total_features[song_id] = song_feat
    with open(f'eball_id_feat_{sub_dir}.pkl', 'wb') as f:
        pkl.dump(total_features, f)

if __name__ == '__main__':
    # getBPM()
    # getTempoFeatures()
    getTempoFeatures_Indv(sys.argv[1])