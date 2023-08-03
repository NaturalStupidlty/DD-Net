import numpy as np
import scipy.ndimage.interpolation as inter

from scipy.signal import medfilt
from scipy.spatial.distance import cdist


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
