import numpy as np
import scipy
import librosa
import glob
import matplotlib.pyplot as plt

path_to_model = 'model_50000iter_003rate_6depth.cbm'
test_path = '/home/evgeny/data/ml_sum_school/data_v_7_stc/test/'
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
feat_map_test = read_all_and_calc_feat(test_path + '*.wav')

## Read metadata.


# Form features and labels vectors.
# Labels: 0 - background, 1 - bags, 2 - door, 3 - keyboard, 4 - knocking_door, 5 - ring, 6 - speech, 7 - tool;
lbl_map = ['background', 'bags', 'door', 'keyboard', 'knocking_door', 'ring', 'speech', 'tool']#, 'unknown']
lbl_map = ['unknown']
lbl_map = ['background', 'bags', 'door', 'keyboard', 'knocking_door', 'ring', 'speech', 'tool', 'unknown']
# test_data = []
# for feat in feat_map_test:
#     if feat[0].contains('background')
#         labels.append('background')
#     if feat[0].contains('bags')
#         labels.append('bags')
#     if feat[0].contains('door')
#         labels.append('door')
#     if feat[0].contains('keyboard')
#         labels.append('keyboard')
#     if feat[0].contains('knocking_door')
#         labels.append('knocking_door')
#     if feat[0].contains('ring')
#         labels.append('ring')
#     if feat[0].contains('tool')
#         labels.append('tool')
#     if feat[0].contains('unknown')
#         labels.append('unknown')
#     print(labels[-1])
#     test_data.append(np.hstack((feat[1], feat[2], feat[3], feat[4], feat[5], feat[6], feat[7])))

test_data = []
labels_test = []
for feat in feat_map_test:
    name = feat[0].split('/')[-1]
    for lbl in lbl_map:
        if name.find(lbl) == 0:
            labels_test.append(lbl)
            test_data.append(np.hstack((feat[1], feat[2], feat[3], feat[4], feat[5], feat[6], feat[7])))


##### testing
from catboost import Pool, CatBoostClassifier
from CatBoostClassifier import load_model
from sklearn import preprocessing

model = CatBoostClassifier.load_model(path_model)

le_test = preprocessing.LabelEncoder()
le_test.fit(labels_test)
lbl_test_transf = le_test.transform(labels_test)
preds_test = model.predict(test_data)
preds_proba_test = model.predict_proba(test_data)

# Comparison.
acc_test = np.nonzero(np.transpose(preds_test)[0] - lbl_test_transf)[0].shape[0] / len(labels_test)
acc_test = 1 - acc_test
# Build area under precision/prob(false alarm)
true_prob = []
for i in range(preds.shape[0]):
    if lbl_test_transf[i] == 8:
        true_prob.append(1 - preds_proba_test[i].max())
    else:
        true_prob.append(preds_proba_test[i][lbl_transf[i]])


acc_plot = []
false_alarm_prob = []
preds_th = []
fa_th = []
threshs = np.linspace(0, 1, 101)

for th in threshs:
    for i in range(preds.shape[0]):
        if lbl_transf[i] == 8:
            if preds_proba[i].max() > th:
                preds_th.append(1)
                fa_th.append(1)
            else:
                preds_th.append(0)
        else:
            if preds_proba[i][lbl_transf[i]] > th:
                preds_th.append(1)
            else:
                preds_th.append(0)
            if preds_proba[i][np.array(preds_proba[i]).argmax()] == lbl_transf[i]:
                1
            else:
                if preds_proba[i][np.array(preds_proba[i]).argmax()] > th:
                    fa_th.append(1)
    acc_plot.append(np.array(preds_th).sum() / len(preds_th))
    false_alarm_prob.append(np.array(fa_th).sum())

import matplotlib.pyplot as plt

plt.plot(false_alarm_prob, acc_plot)
plt.show()
