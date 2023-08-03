import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation as inter

from tqdm import tqdm
from scipy.signal import medfilt
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from keras.src.layers import Lambda, Reshape


class Config:
    def __init__(self):
        self.frame_length = 32
        self.joints_number = 22
        self.joints_dimension = 2
        self.features_dimension = 231
        self.classes_number = 14
        self.filters = 16
        self.data_directory = '../data/SHREC/'


def poses_difference(frame):
    height, width = frame.get_shape()[1], frame.get_shape()[2]
    frame = tf.subtract(frame[:, 1:, ...], frame[:, :-1, ...])
    frame = tf.image.resize(frame, size=[height, width])

    return frame


def pose_motion(frames, frame_length):
    difference_slow = Lambda(lambda x: poses_difference(x))(frames)
    difference_slow = Reshape((frame_length, -1))(difference_slow)

    frames = Lambda(lambda x: x[:, ::2, ...])(frames)
    difference_fast = Lambda(lambda x: poses_difference(x))(frames)
    difference_fast = Reshape((int(frame_length / 2), -1))(difference_fast)

    return difference_slow, difference_fast


def zoom_frames(frames, new_frame_length=64, new_joints_number=22, new_joints_dimension=2):
    frame_length = frames.shape[0]
    zoomed_frames = np.empty([new_frame_length, new_joints_number, new_joints_dimension])
    for j in range(new_joints_number):
        for d in range(new_joints_dimension):
            frames[:, j, d] = medfilt(frames[:, j, d], 3)
            zoomed_frames[:, j, d] = inter.zoom(frames[:, j, d], new_frame_length / frame_length)[:new_frame_length]
    return zoomed_frames


def get_joint_collection_distances(frames, config):
    """
    Calculates triangular matrix of pairwise Euclidian distances between joints

    :param frames:
    :param config:
    :return:
    """

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
    """
    Normalizes frames to start point

    :param frames:
    :return:
    """
    frames[:, :, 0] = frames[:, :, 0] - np.mean(frames[:, :, 0])
    frames[:, :, 1] = frames[:, :, 1] - np.mean(frames[:, :, 1])

    return frames


def prepare_data(config, filename: str):
    data = pickle.load(open(config.data_directory + filename, "rb"))

    Labels = []
    Distances = []
    Motion = []

    print(f"Preparing data from {filename}...")

    for i in tqdm(range(len(data['skeleton']))):
        motion = np.copy(data['skeleton'][i]).reshape([-1, config.joints_number, config.joints_dimension])
        motion = zoom_frames(motion, config.frame_length, config.joints_number, config.joints_dimension)
        motion = normalize_range(motion)

        label = np.zeros(config.classes_number)
        label[data['label'][i] - 1] = 1

        distance = get_joint_collection_distances(motion, config)

        Distances.append(distance)
        Motion.append(motion)
        Labels.append(label)

    Distances = np.stack(Distances)
    Motion = np.stack(Motion)
    Labels = np.stack(Labels)

    return [Distances, Motion], Labels


def plot_history(history):
    """
    Plot training & validation metrics

    :param history:
    :return:
    """
    plt.plot(history.history['accuracy'], color='blue')
    plt.plot(history.history['val_accuracy'], color='orange')
    plt.plot(history.history['auc'], color='green')  # Assuming AUC is named 'auc' in training_history

    plt.title('Model Metrics')
    plt.ylabel('Metrics')
    plt.xlabel('Epoch')
    plt.legend(['Train Accuracy', 'Validation Accuracy', 'ROC AUC'], loc='upper left')
    plt.show()


def plot_confusion_matrix(predictions, labels):
    matrix = confusion_matrix(np.argmax(labels, axis=1), np.argmax(predictions, axis=1))
    plt.figure(figsize=(10, 10))
    plt.imshow(matrix)
    plt.show()
