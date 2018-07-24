import numpy as np
import scipy
import librosa
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# train_path = '/home/evgeny/data/ml_sum_school/data_v_7_stc/audio_tmp/'
# train_path = '/home/evgeny/data/ml_sum_school/data_v_7_stc/audio_tmp_train/'
train_path = '/home/evgeny/data/ml_sum_school/data_v_7_stc/audio/'
# test_path = '/home/evgeny/data/ml_sum_school/data_v_7_stc/audio_tmp_test/'
meta_filetrain_path = '/home/evgeny/data/ml_sum_school/data_v_7_stc/meta/meta.txt'

def norm(x):
    if x.max() - x.min() == 0:
        return x
    x = x / (x.max() - x.min())
    return (x - x.min())

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
            chroma_stft = norm(librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=4096))
            chroma_stft_hist = np.histogram(chroma_stft)
            # Feat #2  CQT hist.
            chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_cq_hist = np.histogram(norm(chroma_cq))
            # Feat #3 Gaussian mixtures centroid.
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, S=None, n_fft=2048, hop_length=chroma_stft.shape[1], freq=None)
            spectral_centroid_hist = np.histogram(norm(spectral_centroid))
            # Feat #4 Tempo, beat count, beats per length.
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr) # Feat
            # Feat #5 Zero crossing rate.
            zcr = np.nonzero(librosa.zero_crossings(y))
            # Tonnetz
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr, chroma=None)
            tonnetz_hist = np.histogram(norm(tonnetz))
            # Separate harmonics and percussives into two waveforms
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            y_harm_hist = np.histogram(norm(y_harmonic))
            y_perc_hist = np.histogram(norm(y_percussive))
            # Tempogram.
            # librosa.feature.tempogram()
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            mfcc_hist = np.histogram(mfcc)
            #d_mfcc=librosa.feature.delta(norm(mfcc))
            #d_mfcc_hist = np.histogram(d_mfcc)
            # Hist stft
            # delta_stft = norm(librosa.feature.delta(chroma_stft))
            # d_stft_hist = np.histogram(delta_stft)

            mel = norm(librosa.feature.melspectrogram(y=y, sr=sr))
            mel_hist = np.histogram(mel)
            # Onesets detection.
            o_env = librosa.onset.onset_strength(y, sr=sr)
            # times = librosa.frames_to_time(np.arange(len(o_env)), sr=sr)
            onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
            os=[]
            for o in o_env:
                if o > 0.5:
                    os.append(o)

            feat_map.append([file, tempo, zcr[0].shape[0], beat_frames.shape[0], beat_frames.shape[0]*1e5/y.shape[0], o_env.shape[0], len(os),
                            chroma_stft_hist[0], chroma_cq_hist[0], spectral_centroid_hist[0], tonnetz_hist[0], y_harm_hist[0], y_perc_hist[0], mfcc_hist[0], mel_hist[0]])
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
        if y is not None:
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
feat_map = read_all_and_calc_feat(train_path + '*.wav')
#feat_map_test = read_all_and_calc_feat(test_path + '*.wav')
# feat_map = read_all_and_calc_feat_and_disp(train_path + '*.wav')

## Read metadata.
f = open(meta_filetrain_path)
meta_dict = {}
for line in f:
    line = line.split()
    filename = line[0]
    label_str = line[4]
    meta_dict[filename] = label_str

## Prepare training set

# Form features and labels vectors.
# Labels: 0 - background, 1 - bags, 2 - door, 3 - keyboard, 4 - knocking_door, 5 - ring, 6 - speech, 7 - tool;
train_data = []
labels = []
for feat in feat_map:
    label = meta_dict[feat[0].split('/')[-1]]
    print(label)
    train_data.append(np.hstack((feat[1], feat[2], feat[3], feat[4], feat[5], feat[6], feat[7], feat[8], feat[9], feat[10], feat[11], feat[12], feat[13], feat[14])))
    labels.append(label)

# Read testing set.
# test_data = []
# labels_test = []
# for feat in feat_map_test:
#     label = meta_dict[feat[0].split('/')[-1]]
#     print(label)
#     test_data.append(np.hstack((feat[1], feat[2], feat[3], feat[4], feat[5], feat[6], feat[7], feat[8])))
#     labels_test.append(label)

############################################
#         Training                         #
############################################

from catboost import Pool, CatBoostClassifier
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(labels)
lbl_transf = le.transform(labels)

# Split test/train set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data, lbl_transf, test_size=0.1, random_state=42)

# Initialize CatBoostClassifier
# model = CatBoostClassifier(iterations=2, learning_rate=1, depth=2, loss_function='MultiClass')
model = CatBoostClassifier(iterations=5000, learning_rate=0.03, depth=6, loss_function='MultiClass')

# Fit model
model.fit(X_train, y_train)

############################################
#         Testing                          #
############################################
#lbl_test = le.transform(y_test)
preds = model.predict(X_test)
preds_proba = model.predict_proba(X_test)

# Comparison.
acc = np.nonzero(np.transpose(preds)[0] - y_test)[0].shape[0] / y_test.shape[0]

# save results
catboost.CatBoostClassifier.save_model(model, 'model_50000iter_rate_003_depth7.cbm')
