import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation as inter


from tqdm import tqdm
from tensorflow import Tensor
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


def poses_difference(frame: Tensor) -> Tensor:
    """
    Calculate the difference between consecutive frames' joint positions.

    :param frame: (Tensor) - Input tensor of shape (batch_size, frame_length, joints_number, joints_dimension).

    :return: (Tensor) - Difference tensor of the same shape as the input frame.
    """
    height, width = frame.get_shape()[1], frame.get_shape()[2]
    frame = tf.subtract(frame[:, 1:, ...], frame[:, :-1, ...])
    frame = tf.image.resize(frame, size=[height, width])
    return frame


def pose_motion(frames: Tensor, frame_length: int) -> (Tensor, Tensor):
    """
    Extract slow and fast motion differences from input frames.

    :param frames: (Tensor) - Input tensor of shape (batch_size, frame_length, joints_number, joints_dimension).
    :param frame_length: (int) - Number of frames in each sequence.

    :return: (Tensor, Tensor) - Tuple containing two tensors: slow motion differences (frame_length, features) and fast motion differences (frame_length/2, features).
    """
    difference_slow = Lambda(lambda x: poses_difference(x))(frames)
    difference_slow = Reshape((frame_length, -1))(difference_slow)

    frames = Lambda(lambda x: x[:, ::2, ...])(frames)
    difference_fast = Lambda(lambda x: poses_difference(x))(frames)
    difference_fast = Reshape((int(frame_length / 2), -1))(difference_fast)

    return difference_slow, difference_fast


def zoom_frames(frames: np.ndarray, new_frame_length: int = 64, new_joints_number: int = 22,
                new_joints_dimension: int = 2) -> np.ndarray:
    """
    Resize input frames to a new frame length while preserving joint positions.

    :param frames: (numpy.ndarray) - Input frames of shape (frame_length, joints_number, joints_dimension).
    :param new_frame_length: (int) - New desired frame length.
    :param new_joints_number: (int) - New desired number of joints.
    :param new_joints_dimension: (int) - New desired dimension of joints.

    :return: (numpy.ndarray) - Zoomed frames of shape (new_frame_length, new_joints_number, new_joints_dimension).
    """
    frame_length = frames.shape[0]
    zoomed_frames = np.empty([new_frame_length, new_joints_number, new_joints_dimension])
    for j in range(new_joints_number):
        for d in range(new_joints_dimension):
            frames[:, j, d] = medfilt(frames[:, j, d], 3)
            zoomed_frames[:, j, d] = inter.zoom(frames[:, j, d], new_frame_length / frame_length)[:new_frame_length]
    return zoomed_frames


def get_joint_collection_distances(frames, config) -> np.ndarray:
    """
    Calculate pairwise Euclidean distances between joints for each frame.

    :param frames: (numpy.ndarray) - Input frames of shape (frame_length, joints_number, joints_dimension).
    :param config: (Config) - Configuration object containing parameters.

    :return: (numpy.ndarray) - Triangular matrix of pairwise distances of shape (frame_length, joints_number*(joints_number-1)/2).
    """
    pairwise_distances_list = []
    upper_triangle_indices = np.triu_indices(config.joints_number, k=1)

    for frame in range(config.frame_length):
        current_frame_points = frames[frame]
        distances_matrix = cdist(current_frame_points,
                                 np.concatenate([current_frame_points, np.zeros([1, config.joints_dimension])]),
                                 'euclidean')
        pairwise_distances = distances_matrix[upper_triangle_indices]
        pairwise_distances_list.append(pairwise_distances)

    jcd_matrix = np.stack(pairwise_distances_list)
    return jcd_matrix


def normalize_range(frames) -> np.ndarray:
    """
    Normalize joint positions within frames to start from the mean.

    :param frames: (numpy.ndarray) - Input frames of shape (frame_length, joints_number, joints_dimension).

    :return: (numpy.ndarray) - Normalized frames with joint positions starting from the mean, same shape as input frames.
    """
    frames[:, :, 0] = frames[:, :, 0] - np.mean(frames[:, :, 0])
    frames[:, :, 1] = frames[:, :, 1] - np.mean(frames[:, :, 1])
    return frames


def prepare_data(config, filename: str) -> (list, np.ndarray):
    """
    Prepare data from a given file.

    :param config: (Config) - Configuration object containing parameters.
    :param filename: (str) - Name of the data file.

    :return: (list, numpy.ndarray) - List containing distance and motion data, and ndarray containing labels.
    """
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


def plot_history(history) -> None:
    """
    Plot training & validation metrics.

    :param history: (History) - Training history returned by the model's fit method.
    """
    plt.plot(history.history['accuracy'], color='blue')
    plt.plot(history.history['val_accuracy'], color='orange')
    plt.plot(history.history['auc'], color='green')

    plt.title('Model Metrics')
    plt.ylabel('Metrics')
    plt.xlabel('Epoch')
    plt.legend(['Train Accuracy', 'Validation Accuracy', 'ROC AUC'], loc='upper left')
    plt.show()


def plot_confusion_matrix(predictions, labels) -> None:
    """
    Plot the confusion matrix based on predicted and true labels.

    :param predictions: (ndarray) - Predicted labels.
    :param labels: (ndarray) - True labels.
    """
    matrix = confusion_matrix(np.argmax(labels, axis=1), np.argmax(predictions, axis=1))
    plt.figure(figsize=(10, 10))
    plt.imshow(matrix)
