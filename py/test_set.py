import numpy as np
import scipy
import librosa
import glob
import matplotlib.pyplot as plt

test_path = '/home/evgeny/data/ml_sum_school/data_v_7_stc/audio_tmp_test/'
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
        y = y/(y.max()-y.min())
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
feat_map_test = read_all_and_calc_feat(test_path + '*.wav')

## Read metadata: all unique labels.
f = open(meta_filetrain_path)
meta_dict_unique = {}
for line in f:
    line = line.split()
    label_str = line[4]
    meta_dict_unique[label_str]

# Form features and labels vectors.
# Labels: 0 - background, 1 - bags, 2 - door, 3 - keyboard, 4 - knocking_door, 5 - ring, 6 - speech, 7 - tool;
test_data = []
for feat in feat_map_test:
    test_data.append(np.hstack((feat[1], feat[2], feat[3], feat[4], feat[5], feat[6], feat[7])))


##### testing
from catboost import Pool, CatBoostClassifier
from sklearn import preprocessing

model = catboost.CatBoostClassifier.load_model(path_model)

le = preprocessing.LabelEncoder()
le.fit(labels)
lbl_transf = le.transform(labels)
preds = model.predict(X_test)
preds_proba = model.predict_proba(X_test, path_model)

# Comparison.
# acc = np.nonzero(np.transpose(preds)[0] - y_test)[0].shape[0] / y_test.shape[0]
