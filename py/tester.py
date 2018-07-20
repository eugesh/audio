import numpy as np
import scipy
import glob
import catboost
from catboost import CatBoostClassifier
import librosa
# from feat_extractor import read_all_and_calc_feat

path_to_model = 'model_50000iter_003rate_6depth.cbm'
test_path = '/home/evgeny/data/ml_sum_school/data_v_7_stc/audio_tmp_test/'

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

feat_map_test = read_all_and_calc_feat(test_path + '*.wav')

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
for feat in feat_map_test:
    label = meta_dict[feat[0].split('/')[-1]]
    print(label)
    train_data.append(np.hstack((feat[1], feat[2], feat[3], feat[4], feat[5], feat[6], feat[7])))
    labels.append(label)

model_test = catboost.load_model()


from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(labels)
lbl_transf = le.transform(labels)

# Split test/train set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data, lbl_transf, test_size=0.9, random_state=42)

preds = model_test.predict(X_test)
preds_proba = model_test.predict_proba(X_test)

# Comparison.
acc = np.nonzero(np.transpose(preds)[0] - y_test)[0].shape[0] / y_test.shape[0]
