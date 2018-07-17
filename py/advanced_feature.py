import numpy as np
import scipy
import librosa
import glob
import matplotlib.pyplot as plt

train_path = '/home/evgeny/data/ml_sum_school/data_v_7_stc/audio_tmp/'
meta_filetrain_path = '/home/evgeny/data/ml_sum_school/data_v_7_stc/meta/meta.txt'

def read_all_and_calc_feat(dirname):
    '''
    Reads all audio files and calculates features.
    '''
    vec_of_chroma_stft = []
    vec_of_chroma_cq = []
    feat_map = []
    files=glob.glob(dirname)
    for file in files:
        print("file = ", file)
        y, sr = librosa.load(file)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=4096)
        print(chroma_stft.shape[1])
        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=chroma_stft.shape[1])
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, S=None, n_fft=2048, hop_length=chroma_stft.shape[1], freq=None)
        # tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        feat_map.append([file, chroma_cq, spectral_centroid])
    return feat_map

def read_all_and_calc_feat_and_disp(dirname):
    '''
    Reads all audio files and calculates features.
    '''
    vec_of_chroma_stft = []
    vec_of_chroma_cq = []
    feat_map = []
    files=glob.glob(dirname)
    for file in files:
        print("file = ", file)
        y, sr = librosa.load(file)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=4096)
        print(chroma_stft.shape[1])
        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, S=None, n_fft=2048, hop_length=chroma_stft.shape[1], freq=None)
        # tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        feat_map.append([file, chroma_cq, chroma_stft])
        plt.figure(figsize=(6, 6))
        ax = plt.subplot(3, 1, 1)
        librosa.display.specshow(librosa.power_to_db(chroma_stft, ref=np.max),
                                 y_axis='mel', x_axis='time')
        plt.title('chroma_stft')
        ax = plt.subplot(3, 1, 2)
        librosa.display.specshow(librosa.power_to_db(chroma_cq, ref=np.max),
                                 y_axis='mel', x_axis='time')
        plt.title('chroma_cq')
        ax = plt.subplot(3, 1, 3)
        librosa.display.specshow(librosa.power_to_db(spectral_centroid, ref=np.max),
                                 y_axis='mel', x_axis='time')
        plt.title('spectral_centroid')
        plt.show()
    return feat_map

#####################################################################################

## Load all audio files and calculate features.
feat_map = read_all_and_calc_feat_and_disp(train_path + '*.wav')

## Read metadata.
f = open(meta_filetrain_path)
meta_dict = {}
for line in f:
    line = line.split()
    filename = line[0]
    label_str = line[4]
    meta_dict[filename] = label_str

# Form features and labels vectors.
# Labels: 0 - background, 1 - bags, 2 - door, 3 - keyboard, 4 - knocking_door, 5 - ring, 6 - speech, 7 - tool;
train_data = []
labels = []
for feat in feat_map:
    label = meta_dict[feat[0].split('/')[-1]]
    print(label)
    train_data.append(np.vstack((feat[1], feat[2])))
    labels.append(label)



# Train classifier.
from catboost import Pool, CatBoostClassifier
