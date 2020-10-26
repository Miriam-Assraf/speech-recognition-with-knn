import numpy as np


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def get_most_common(arr):
    unique, pos = np.unique(arr, return_inverse=True)  # get all unique elements and their positions
    counts = np.bincount(pos)  # count the number of each unique element
    max_count_pos = counts.argmax()  # get the positions of the maximum count
    # return the element with max counts
    return unique[max_count_pos]


class KNN:
    def __init__(self, k, x, y):
        self.k = k
        # fit training data and labels
        self.x_train = x
        self.y_train = y

    # predict for a new test data
    def predict(self, x):
        # calculate all distances
        distances = [euclidean_distance(x, x_train) for x_train in
                     self.x_train]  # list comprehension- all distances from new sample x to each sample in X
        # get k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]  # sort distances and get indexes of k nearest ones
        k_nearest_labels = [self.y_train[i] for i in k_indices]  # get labels of nearest neighbors found by their index
        # majority vote- return most common label
        return get_most_common(k_nearest_labels)

    # predict new given test data set
    def predict_all(self, x):
        return [self.predict(xi) for xi in x]  # predict for each data xi in given data set x
