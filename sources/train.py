import random

from sources.utils import Config, prepare_data, plot_history, plot_confusion_matrix
from sources.DDNet import DDNet


def main():
    random.seed(69)

    config = Config()

    DD_Net = DDNet(config)

    X_train, Y_train = prepare_data(config, "train.pkl")
    X_valid, Y_valid = prepare_data(config, "valid.pkl")

    history = DD_Net.train(X_train, Y_train, X_valid, Y_valid, learning_rate=1e-3, epochs=300)

    DD_Net.save('../models/ddnet.h5')

    plot_history(history)

    Y_pred = DD_Net.predict(X_valid)

    plot_confusion_matrix(Y_pred, Y_valid)


if __name__ == "__main__":
    main()
