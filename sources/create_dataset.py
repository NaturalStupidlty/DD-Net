import pickle
import numpy as np
from tqdm import tqdm
from scipy.signal import medfilt

from sources.utils import Config


def load_gesture_files(file_path: str):
    """
    Load gesture files from a given path.

    :param file_path: (str) - Path to the gesture files.
    :return: (numpy.ndarray) - Loaded gesture data as int16.
    """
    return np.loadtxt(file_path).astype('int16')


def load_skeleton_data(skeleton_path: str):
    """
    Load skeleton data from a given path and apply median filtering.

    :param skeleton_path: (str) - Path to the skeleton data file.
    :return: (numpy.ndarray) - Loaded and filtered skeleton data as float32.
    """
    skeleton = np.loadtxt(skeleton_path).astype('float32')
    for j in range(skeleton.shape[1]):
        skeleton[:, j] = medfilt(skeleton[:, j])
    return skeleton


def create_dataset(gestures: np.ndarray, config: Config):
    """
    Create a dataset from the provided gesture information.

    :param gestures: (np.ndarray) - List of gesture information.
    :param config: (Config) - Configuration object.
    :return: (dict) - Dataset containing 'skeleton' and 'label' data.
    """
    dataset = {'skeleton': [], 'label': []}

    for gesture_info in tqdm(gestures):
        idx_gesture, idx_finger, idx_subject, idx_essai, label, _, _ = gesture_info

        skeleton_path = (
            f"{config.data_directory}/gesture_{idx_gesture}/finger_{idx_finger}/"
            f"subject_{idx_subject}/essai_{idx_essai}/skeletons_image.txt"
        )
        skeleton = load_skeleton_data(skeleton_path)

        dataset['skeleton'].append(skeleton)
        dataset['label'].append(label)

    return dataset


def save_dataset(dataset: dict, file_path: str):
    """
    Save a dataset to a file using pickle.

    :param dataset: (dict) - Dataset to be saved.
    :param file_path: (str) - Path to the output file.
    """
    with open(file_path, "wb") as f:
        pickle.dump(dataset, f)


def main():
    """
    Main function to process and save datasets.
    """
    config = Config()
    train_files = load_gesture_files(config.data_directory + 'train_gestures.txt')
    valid_files = load_gesture_files(config.data_directory + 'valid_gestures.txt')

    train_dataset = create_dataset(train_files, config)
    valid_dataset = create_dataset(valid_files, config)

    save_dataset(train_dataset, config.data_directory + "train.pkl")
    save_dataset(valid_dataset, config.data_directory + "valid.pkl")


if __name__ == "__main__":
    main()
