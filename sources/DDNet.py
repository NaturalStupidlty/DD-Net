import os
import time
import numpy as np
from tensorflow import Tensor
from keras.models import Model
from keras.optimizers.legacy import Adam
from keras.src.callbacks import ReduceLROnPlateau, History
from keras.metrics import Precision, Recall, AUC
from keras.layers import Conv1D, BatchNormalization, LeakyReLU, Dropout, Dense, Input, SpatialDropout1D, \
    MaxPooling1D, GlobalMaxPooling1D, concatenate

from sources.utils import Config, pose_motion


class DDNet:
    """
    Implementation of DD-Net model.

    Attributes:
        model: (Model) - The built Keras model.
        model_config: (Config) - Configuration settings for the model.

    Methods:
        dense1D(x, filters)
        convolution1D(x, filters, kernel)
        convolutions_block(x, filters)
        build_model_block(x, filters, max_pooling)
        build_jcd_block(x, filters)
        build_slow_motion_block(x, filters)
        build_fast_motion_block(x, filters)
        build_main_block(x, filters)
        build_output_block(x, classes_number)
        build_model()
        train(x_train, y_train, x_valid, y_valid, learning_rate, epochs)
        predict(data, verbose)
        save(save_path)
    """
    def __init__(self, model_config: Config, verbose: bool = True) -> None:
        """
        Builds the model.

        :param model_config: (Config) - Configuration settings for the model.
        :param verbose: (bool) - If True, print model summary after building.
        """
        self.model = None
        self.model_config = model_config
        self.build_model()

        if verbose and self.model is not None:
            print(self.model.summary())

    @staticmethod
    def dense1D(x: Tensor, filters: int) -> Tensor:
        """
        Apply a 1D dense layer with batch normalization and LeakyReLU activation.

        :param x: (Tensor) - Input tensor.
        :param filters: (int) - Number of filters (units) in the dense layer.
        :return: (Tensor) - Output tensor after applying dense layer.
        """
        x = Dense(filters, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    @staticmethod
    def convolution1D(x: Tensor, filters: int, kernel: int) -> Tensor:
        """
        Apply a 1D convolutional layer with batch normalization and LeakyReLU activation.

        :param x: (Tensor) - Input tensor.
        :param filters: (int) - Number of filters (output channels) in the convolutional layer.
        :param kernel: (int) - Size of the convolutional kernel.
        :return: (Tensor) - Output tensor after applying convolutional layer.
        """
        x = Conv1D(filters, kernel_size=kernel, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    @staticmethod
    def convolutions_block(x: Tensor, filters: int) -> Tensor:
        """
        Apply a block of two consecutive 1D convolutional layers with batch normalization and LeakyReLU activation.

        :param x: (Tensor) - Input tensor.
        :param filters: (int) - Number of filters (output channels) for each convolutional layer.
        :return: (Tensor) - Output tensor after applying the convolutions block.
        """
        x = DDNet.convolution1D(x, filters, 3)
        x = DDNet.convolution1D(x, filters, 3)
        return x

    @staticmethod
    def build_model_block(x: Tensor, filters: int, max_pooling: bool) -> Tensor:
        """
        Build a model block consisting of multiple layers including convolutions and optional max pooling.
        In DD-Net we do not use max pooling to build fast motion block.

        :param x: (Tensor) - Input tensor.
        :param filters: (int) - Number of filters (output channels) for convolutional layers.
        :param max_pooling: (bool) - Whether to apply max pooling after certain convolutions.
        :return: (Tensor) - Output tensor after applying the model block.
        """
        x = DDNet.convolution1D(x, filters * 2, 1)
        x = SpatialDropout1D(0.1)(x)

        x = DDNet.convolution1D(x, filters, 3)
        x = SpatialDropout1D(0.1)(x)

        x = DDNet.convolution1D(x, filters, 1)
        if max_pooling:
            x = MaxPooling1D(2)(x)
        x = SpatialDropout1D(0.1)(x)
        return x

    @staticmethod
    def build_jcd_block(x: Tensor, filters: int) -> Tensor:
        """
        Build a block specific for Joint Collection Distances (JCD) processing.

        :param x: (Tensor) - Input tensor.
        :param filters: (int) - Number of filters (output channels) for convolutional layers.
        :return: (Tensor) - Output tensor after applying the JCD block.
        """
        x = DDNet.build_model_block(x, filters, max_pooling=True)
        return x

    @staticmethod
    def build_slow_motion_block(x: Tensor, filters: int) -> Tensor:
        """
        Build a block for processing slow motion data.

        :param x: (Tensor) - Input tensor.
        :param filters: (int) - Number of filters (output channels) for convolutional layers.
        :return: (Tensor) - Output tensor after applying the slow motion block.
        """
        x = DDNet.build_model_block(x, filters, max_pooling=True)
        return x

    @staticmethod
    def build_fast_motion_block(x: Tensor, filters: int) -> Tensor:
        """
        Build a block for processing fast motion data.

        :param x: (Tensor) - Input tensor.
        :param filters: (int) - Number of filters (output channels) for convolutional layers.
        :return: (Tensor) - Output tensor after applying the fast motion block.
        """
        x = DDNet.build_model_block(x, filters, max_pooling=False)
        return x

    @staticmethod
    def build_main_block(x: Tensor, filters: int) -> Tensor:
        """
        Build the main block of the model, consisting of multiple convolutional layers with max pooling.

        :param x: (Tensor) - Input tensor.
        :param filters: (int) - Number of filters (output channels) for convolutional layers.
        :return: (Tensor) - Output tensor after applying the main block.
        """
        x = DDNet.convolutions_block(x, filters * 2)
        x = MaxPooling1D(2)(x)
        x = SpatialDropout1D(0.1)(x)

        x = DDNet.convolutions_block(x, filters * 4)
        x = MaxPooling1D(2)(x)
        x = SpatialDropout1D(0.1)(x)

        x = DDNet.convolutions_block(x, filters * 8)
        x = SpatialDropout1D(0.1)(x)
        return x

    @staticmethod
    def build_output_block(x: Tensor, classes_number: int) -> Tensor:
        """
        Build the output block of the model.

        :param x: (Tensor) - Input tensor.
        :param classes_number: (int) - Number of classes for classification.
        :return: (Tensor) - Output tensor after applying the output block.
        """
        x = GlobalMaxPooling1D()(x)
        x = DDNet.dense1D(x, 128)
        x = Dropout(0.5)(x)

        x = DDNet.dense1D(x, 128)
        x = Dropout(0.3)(x)
        x = Dense(classes_number, activation='softmax')(x)
        return x

    def build_model(self) -> None:
        """
        Build the DDNet model architecture from the configurations.

        :return: None
        """
        Distance = Input(name='Distance', shape=(self.model_config.frame_length, self.model_config.features_dimension))
        Motion = Input(name='Motion',
                       shape=(self.model_config.frame_length,
                              self.model_config.joints_number,
                              self.model_config.joints_dimension))

        slow_diff, fast_diff = pose_motion(Motion, self.model_config.frame_length)

        # Joint Collection Distances
        model = self.build_jcd_block(Distance, self.model_config.filters)

        # Cartesian coordinates
        slow_diff = self.build_slow_motion_block(slow_diff, self.model_config.filters)
        fast_diff = self.build_fast_motion_block(fast_diff, self.model_config.filters)

        model = concatenate([model, slow_diff, fast_diff])
        model = self.build_main_block(model, self.model_config.filters)
        model = self.build_output_block(model, self.model_config.classes_number)

        self.model = Model(inputs=[Distance, Motion], outputs=model)

    def train(self,
              x_train: np.ndarray, y_train: np.ndarray,
              x_valid: np.ndarray, y_valid: np.ndarray,
              learning_rate: float = 1e-3,
              epochs: int = 100) -> History:
        """
        Train the DDNet model on the provided data.

        :param x_train: (np.ndarray) - Training input data.
        :param y_train: (np.ndarray) - Training labels.
        :param x_valid: (np.ndarray) - Validation input data.
        :param y_valid: (np.ndarray) - Validation labels.
        :param learning_rate: (float) - Learning rate for the optimizer.
        :param epochs: (int) - Number of training epochs.
        :return: (History) - Training history.
        """
        precision = Precision(name='precision')
        recall = Recall(name='recall')
        roc_auc = AUC(multi_label=True)

        metrics_list = ['accuracy', precision, recall, roc_auc]

        self.model.compile(loss="categorical_crossentropy",
                           optimizer=Adam(learning_rate),
                           metrics=metrics_list)

        callbacks = []
        lr_scheduler = ReduceLROnPlateau(monitor='loss',
                                         factor=0.5,
                                         patience=5,
                                         cooldown=5,
                                         min_lr=5e-6)
        callbacks.append(lr_scheduler)

        history = self.model.fit(x_train,
                                 y_train,
                                 batch_size=len(y_train),
                                 epochs=epochs,
                                 verbose=True,
                                 shuffle=True,
                                 callbacks=callbacks,
                                 validation_data=(x_valid, y_valid)
                                 )

        return history

    def predict(self, data: list[np.ndarray], verbose: bool = True):
        """
        Make predictions using the trained model.

        :param data: (list[np.ndarray]) - List of input data arrays for prediction.
        :param verbose: (bool) - If True, print prediction time.
        :return: (np.ndarray) - Predicted labels.
        """
        start_time = time.time()
        predictions = self.model.predict(data)
        end_time = time.time() - start_time

        if verbose:
            print(f"Prediction time: {end_time} seconds.")

        return predictions

    def save(self, save_path: str):
        """
        Save the model weights to the specified path.

        :param save_path: (str) - Path to save the model weights.
        """
        self.model.save_weights(save_path)

    def load(self, load_path: str):
        """
        Load the model weights from the specified path. It also checks if the model is an .h5 file that exists.

        :param load_path: (str) - Path to load the model weights.
        """
        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        if not load_path.endswith(".h5"):
            raise ValueError(f"Invalid file type: {load_path}. Expected .h5 file.")

        self.model.load_weights(load_path)

