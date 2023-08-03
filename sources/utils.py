import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation as inter

from scipy.signal import medfilt
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix


class Config:
    def __init__(self):
        self.frame_length = 32
        self.joints_number = 22
        self.joints_dimension = 3
        self.classes_number = 14
        self.features_dimension = 231
        self.filters = 16
        self.data_directory = '../data/SHREC/'


def zoom_frames(frames, new_frame_length=64, new_joints_number=25, new_joints_dimension=3):
    frame_length = frames.shape[0]
    zoomed_frames = np.empty([new_frame_length, new_joints_number, new_joints_dimension])
    for j in range(new_joints_number):
        for d in range(new_joints_dimension):
            frames[:, j, d] = medfilt(frames[:, j, d], 3)
            zoomed_frames[:, j, d] = inter.zoom(frames[:, j, d], new_frame_length / frame_length)[:new_frame_length]
    return zoomed_frames


def get_joint_collection_distances(frames, config):
    pairwise_distances_list = []
    upper_triangle_indices = np.triu_indices(config.joints_number, k=1)

    for frame in range(config.frame_length):
        # Calculate pairwise Euclidean distances for the current frame
        current_frame_points = frames[frame]
        distances_matrix = cdist(current_frame_points,
                                 np.concatenate([current_frame_points, np.zeros([1, config.joints_dimension])]),
                                 'euclidean')
        pairwise_distances = distances_matrix[upper_triangle_indices]

        # Append the pairwise distances to the list
        pairwise_distances_list.append(pairwise_distances)

    # Convert the list of pairwise distances into a 3D NumPy array
    jcd_matrix = np.stack(pairwise_distances_list)

    return jcd_matrix


def normalize_range(frames):
    # Normalize to start point
    frames[:, :, 0] = frames[:, :, 0] - np.mean(frames[:, :, 0])
    frames[:, :, 1] = frames[:, :, 1] - np.mean(frames[:, :, 1])
    frames[:, :, 2] = frames[:, :, 2] - np.mean(frames[:, :, 2])

    return frames


def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figure_size=(8, 8)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (samples_number,)
      y_pred:    prediction of the data, with shape (samples_number,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (classes_number,).
      ymap:      dict: any -> string, length == classes_number.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figure_size:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                #annot[i, j] = '%.1f%%\n%d/%d' % (frames, c, s)
                annot[i, j] = '%.1f' % (p)
            elif c == 0:
                annot[i, j] = ''
            else:
                #annot[i, j] = '%.1f%%\n%d' % (frames, c)
                annot[i, j] = '%.1f' % (p)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figure_size)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cbar=False, cmap="YlGnBu")
    plt.savefig(filename)
