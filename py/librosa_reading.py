import numpy as np
import scipy
import librosa
import glob
import matplotlib.pyplot as plt
# https://hi-tech.mail.ru/news/robot-tarakan/?frommail=1

train_path = '/home/evgeny/data/ml_sum_school/data_v_7_stc/audio/'
meta_filetrain_path = '/home/evgeny/data/ml_sum_school/data_v_7_stc/meta/meta.txt'

def read_all_images(dirname):
    '''
    Function for reading all files from dirname.
    '''
    files=glob.glob(dirname)
    vec_of_y = []
    vec_of_sr = []
    for file in files:
        # load(train_path[, sr, mono, offset, duration, …])
        y, sr = librosa.load(file)
        vec_of_y.append(y)
        vec_of_sr.append(sr)
    return (vec_of_y, vec_of_sr)

def calc_features(vec_of_y, vec_of_sr):
    vec_of_chroma_stft = []
    vec_of_chroma_cq = []
    for (y, sr) in (vec_of_y, vec_of_sr):
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=4096)
        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
        vec_of_chroma_stft.append(chroma_stft)
        vec_of_chroma_cq.append(chroma_cr)
    return vec_of_chroma_stft, vec_of_chroma_cq

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
        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
        # vec_of_chroma_stft.append(chroma_stft)
        # vec_of_chroma_cq.append(chroma_cq)
    #return vec_of_chroma_stft, vec_of_chroma_cq
        feat_map.append([file, chroma_stft, chroma_cq])
    return feat_map

#load(train_path, sr=22050, mono=True, offset=0.0, duration=None, dtype=<class 'numpy.float32'>, res_type='kaiser_best')
# y, sr = librosa.load(librosa.util.example_audio_file())

# Load all audio files.
# y, sr = librosa.load(train_path + filename)
# vec_of_chroma_stft, vec_of_chroma_cq = read_all_and_calc_feat(train_path + '*.wav')
feat_map = read_all_and_calc_feat(train_path + '*.wav')


# Read metadata
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



# Features visualization.
# plt.figure()
# plt.subplot(2, 1, 1)
# #librosa.display.specshow(chroma_stft, y_axis='chroma')
#
# plt.title('chroma_stft')
# plt.colorbar()
# plt.subplot(2,1,2)
# #librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time')
# plt.title('chroma_cqt')
# plt.colorbar()
# plt.tight_layout()
