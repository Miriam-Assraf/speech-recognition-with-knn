import numpy as np
import glob
import librosa
from scipy import stats
import knn


# convert string label to int
def string_to_int(s):
    nums = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    if s in nums:
        return str(nums[s])


# load audio files from given path (training data) with librosa
# for each example xi get audio features with mfcc and calculate its z-score
# returns a 3D array that contains for each file the z-score array of its features
def get_features(x_path_list):
    x = np.empty(0)
    for num_examples in range(len(x_path_list)):
        y, sr = librosa.load(x_path_list[num_examples], sr=None)
        xi = librosa.feature.mfcc(y=y, sr=sr)
        xi = stats.zscore(xi, axis=1)
        x = np.append(x.reshape(num_examples, xi.shape[0], xi.shape[1]), [xi], axis=0)
    return x


x_train_path = glob.glob('train_data/*/*.wav')  # get train data files path
x_test_path = glob.glob('test_files/*.wav')  # get test data files path

# create empty array for labels at size of number of examples in train data
y_train = np.empty(len(x_train_path), dtype=list)

# get train data labels
for label in range(len(y_train)):
    y_train[label] = x_train_path[label][11:-22]  # the name of folder position- name of folder is the label

x_train = get_features(x_train_path)
x_test = get_features(x_test_path)

# get k=1 nearest neighbors and predict given data set-x_test
one_nn = knn.KNN(1, x_train, y_train)
predictions = one_nn.predict_all(x_test)

# enter to predictions test file name (by position) and predicted number (not label)
for label in range(len(predictions)):
    predictions[label] = x_test_path[label][11:] + " - " + string_to_int(predictions[label])

# save predictions to file
np.savetxt('predictions.txt', [predictions], fmt="%s", delimiter='\n')
