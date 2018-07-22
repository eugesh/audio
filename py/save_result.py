import numpy as np
import scipy
import librosa
import glob
import matplotlib.pyplot as plt


test_path = '/home/evgeny/data/ml_sum_school/data_v_7_stc/test/'
meta_filetrain_path = '/home/evgeny/data/ml_sum_school/data_v_7_stc/meta/meta.txt'
path_model = 'model_50000iter_rate_003_depth6.cbm'


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
        # Read file.
        y, sr = librosa.load(file)
        if y is not None:
            # Feat #1 - Chromagram's hist.
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=4096)
            chroma_stft_hist = np.histogram(chroma_stft)
            # Feat #2  CQT hist.
            chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_cq_hist = np.histogram(chroma_cq)
            # Feat #3 Gaussian mixtures centroid.
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, S=None, n_fft=2048, hop_length=chroma_stft.shape[1], freq=None)
            spectral_centroid_hist = np.histogram(spectral_centroid)
            # Feat #4 Tempo, beat count, beats per length.
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr) # Feat
            # Feat #5 Zero crossing rate.
            zcr = np.nonzero(librosa.zero_crossings(y))

            feat_map.append([file, tempo, zcr[0].shape[0], beat_frames.shape[0], beat_frames.shape[0]*1e5/y.shape[0],
                            chroma_stft_hist[0], chroma_cq_hist[0], spectral_centroid_hist[0]])
    return feat_map


## Load all audio files and calculate features.
feat_map_test_res = read_all_and_calc_feat(test_path + '*.wav')

test_data_res = []
for feat in feat_map_test_res:
    test_data_res.append(np.hstack((feat[1], feat[2], feat[3], feat[4], feat[5], feat[6], feat[7])))

# lbl_map_test_res = ['background', 'bags', 'door', 'keyboard', 'knocking_door', 'ring', 'speech', 'tool', 'unknown']

##### testing
import catboost
from catboost import Pool, CatBoostClassifier
from sklearn import preprocessing

# model = catboost.CatBoostClassifier.load_model(path_model)

preds_test_res = model.predict(test_data_res)
preds_proba_test_res = model.predict_proba(test_data_res)

# Save in file.
i=0
with open('result.txt', 'w') as f:
    for feat in feat_map_test_res:
        name = feat[0].split('/')[-1]
        lbl = np.int(preds_test_res[i][0])
        print(name, preds_proba_test_res[i][lbl], le.classes_[lbl],file=f)
        i = i + 1
