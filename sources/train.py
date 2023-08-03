import time
import random
import keras
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from keras.models import Model
from keras.src.layers import Conv1D, BatchNormalization, LeakyReLU, Dropout, Dense, Input, SpatialDropout1D,\
    MaxPooling1D, GlobalMaxPooling1D, concatenate

from sources.utils import Config, zoom_frames, normalize_range, get_joint_collection_distances, pose_motion


def convolution1D(x, filters, kernel):
    x = Conv1D(filters, kernel_size=kernel, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    return x


def convolutions_block(x, filters):
    x = convolution1D(x, filters, 3)
    x = convolution1D(x, filters, 3)

    return x


def dense1D(x, filters):
    x = Dense(filters, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    return x


def build_model_block(x, filters, max_pooling: bool):
    x = convolution1D(x, filters * 2, 1)
    x = SpatialDropout1D(0.1)(x)

    x = convolution1D(x, filters, 3)
    x = SpatialDropout1D(0.1)(x)

    x = convolution1D(x, filters, 1)
    if max_pooling:
        x = MaxPooling1D(2)(x)
    x = SpatialDropout1D(0.1)(x)

    return x


def build_jcd_block(x, filters):
    x = build_model_block(x, filters, max_pooling=True)

    return x


def build_slow_motion_block(x, filters):
    x = build_model_block(x, filters, max_pooling=True)

    return x


def build_fast_motion_block(x, filters):
    x = build_model_block(x, filters, max_pooling=False)

    return x


def build_main_block(x, filters):
    x = convolutions_block(x, filters * 2)
    x = MaxPooling1D(2)(x)
    x = SpatialDropout1D(0.1)(x)

    x = convolutions_block(x, filters * 4)
    x = MaxPooling1D(2)(x)
    x = SpatialDropout1D(0.1)(x)

    x = convolutions_block(x, filters * 8)
    x = SpatialDropout1D(0.1)(x)

    return x


def build_output_block(x, classes_number):
    x = GlobalMaxPooling1D()(x)

    x = dense1D(x, 128)
    x = Dropout(0.5)(x)
    x = dense1D(x, 128)
    x = Dropout(0.3)(x)
    x = Dense(classes_number, activation='softmax')(x)

    return x


def build_DD_Net(model_config):
    Distance = Input(name='Distance', shape=(model_config.frame_length, model_config.features_dimension))
    Motion = Input(name='Motion',
                   shape=(model_config.frame_length, model_config.joints_number, model_config.joints_dimension))

    slow_diff, fast_diff = pose_motion(Motion, model_config.frame_length)

    # Joint Collection Distances
    model = build_jcd_block(Distance, model_config.filters)

    # Cartesian coordinates
    slow_diff = build_slow_motion_block(slow_diff, model_config.filters)
    fast_diff = build_fast_motion_block(fast_diff, model_config.filters)

    model = concatenate([model, slow_diff, fast_diff])
    model = build_main_block(model, model_config.filters)
    model = build_output_block(model, model_config.classes_number)

    model = Model(inputs=[Distance, Motion], outputs=model)

    return model


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


def train(model, X_train, Y_train, X_valid, Y_valid, epochs=100):
    callbacks = []
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                     factor=0.5,
                                                     patience=5,
                                                     cooldown=5,
                                                     min_lr=5e-6)
    callbacks.append(lr_scheduler)

    history = model.fit(X_train,
                        Y_train,
                        batch_size=len(Y_train),
                        epochs=epochs,
                        verbose=True,
                        shuffle=True,
                        callbacks=callbacks,
                        validation_data=(X_valid, Y_valid)
                        )

    return model, history


def main():
    random.seed(69)

    config = Config()

    DD_Net = build_DD_Net(config)

    print(DD_Net.summary())

    X_train, Y_train = prepare_data(config, "train.pkl")
    X_valid, Y_valid = prepare_data(config, "valid.pkl")

    lr = 1e-3
    DD_Net.compile(loss="categorical_crossentropy",
                   optimizer=keras.optimizers.legacy.Adam(lr),
                   metrics=['accuracy'])

    DD_Net, training_history = train(DD_Net, X_train, Y_train, X_valid, Y_valid, epochs=300)

    DD_Net.save_weights('../models/ddnet.h5')

    # Plot training & validation accuracy values
    plt.plot(training_history.history['accuracy'])
    plt.plot(training_history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    start_time = time.time()
    Y_pred = DD_Net.predict(X_valid)
    time.time() - start_time

    cnf_matrix = confusion_matrix(np.argmax(Y_valid, axis=1), np.argmax(Y_pred, axis=1))
    plt.figure(figsize=(10, 10))
    plt.imshow(cnf_matrix)
    plt.show()


if __name__ == "__main__":
    main()
