import librosa
import numpy as np
import pickle as pkl
import os
import tqdm

import warnings
warnings.filterwarnings('ignore')

def getBPM():
    bpm_dict = {}
    DIR_PATH = 'Datasets/ballroom/BallroomAnnotations/ballroomGroundTruth/'
    for song in os.listdir(DIR_PATH):
        song_path = DIR_PATH + song
        song_id, song_ext = os.path.splitext(song)
        if(song_ext == '.bpm'):
            with open(song_path, 'r') as f:
                cbpm = f.read().rstrip()
            bpm_dict[song_id] = cbpm
    with open('ball_id_bpm.csv', 'w') as f:
        for sid, bpm in bpm_dict.items():
            f.write(sid + ',' + bpm + '\n')

def getMelSpec(path):
    raw_wave, sample_rate = librosa.load(path, sr=11025)
    win_len = 1024
    data = librosa.feature.melspectrogram(y=raw_wave, sr=sample_rate, 
                                          n_fft=win_len, hop_length=(win_len // 2), 
                                          power=1, n_mels=40, fmin=20, fmax=5000)
    data = np.reshape(data, (data.shape[0], data.shape[1], 1))
    return data.astype(np.float16)

def getTempoFeatures():
    DIR_PATH = 'Datasets/ballroom/BallroomData/'
    total_features = {}
    with open(DIR_PATH + 'allBallroomFiles', 'r') as f:
        fnames = f.readlines()
    for spath in tqdm.tqdm(fnames):
        song_path = DIR_PATH + spath[2: -1]
        song = os.path.basename(song_path)
        song_id, song_ext = os.path.splitext(song)
        if(song_ext == '.wav'):
            song_feat = getMelSpec(song_path)
            total_features[song_id] = song_feat
    with open('ball_id_feat.pkl', 'wb') as f:
        pkl.dump(total_features, f)

if __name__ == '__main__':
    # getBPM()
    getTempoFeatures()