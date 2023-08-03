import random
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from sources.utils import Config, zoom_frames, normalize_range, get_joint_collection_distances
from sources.DDNet import DDNet


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


def main():
    random.seed(69)

    config = Config()

    DD_Net = DDNet(config)

    print(DD_Net.model.summary())

    X_train, Y_train = prepare_data(config, "train.pkl")
    X_valid, Y_valid = prepare_data(config, "valid.pkl")

    history = DD_Net.train(X_train, Y_train, X_valid, Y_valid, learning_rate=1e-3, epochs=30)

    DD_Net.save('../models/ddnet.h5')

    plot_history(history)

    Y_pred = DD_Net.predict(X_valid)

    cnf_matrix = confusion_matrix(np.argmax(Y_valid, axis=1), np.argmax(Y_pred, axis=1))
    plt.figure(figsize=(10, 10))
    plt.imshow(cnf_matrix)
    plt.show()


if __name__ == "__main__":
    main()
