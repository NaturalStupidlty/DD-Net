import time
import numpy as np
from keras.models import Model
from keras.src.callbacks import ReduceLROnPlateau
from keras.optimizers.legacy import Adam
from keras.layers import Conv1D, BatchNormalization, LeakyReLU, Dropout, Dense, Input, SpatialDropout1D, \
    MaxPooling1D, GlobalMaxPooling1D, concatenate

from sources.utils import pose_motion


class DDNet:
    def __init__(self, model_config):
        self.model = None
        self.model_config = model_config
        self.build_model()

    @staticmethod
    def dense1D(x, filters):
        x = Dense(filters, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    @staticmethod
    def convolution1D(x, filters, kernel):
        x = Conv1D(filters, kernel_size=kernel, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    @staticmethod
    def convolutions_block(x, filters):
        x = DDNet.convolution1D(x, filters, 3)
        x = DDNet.convolution1D(x, filters, 3)
        return x

    @staticmethod
    def build_model_block(x, filters, max_pooling: bool):
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
    def build_jcd_block(x, filters):
        x = DDNet.build_model_block(x, filters, max_pooling=True)
        return x

    @staticmethod
    def build_slow_motion_block(x, filters):
        x = DDNet.build_model_block(x, filters, max_pooling=True)
        return x

    @staticmethod
    def build_fast_motion_block(x, filters):
        x = DDNet.build_model_block(x, filters, max_pooling=False)
        return x

    @staticmethod
    def build_main_block(x, filters):
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
    def build_output_block(x, classes_number):
        x = GlobalMaxPooling1D()(x)
        x = DDNet.dense1D(x, 128)
        x = Dropout(0.5)(x)

        x = DDNet.dense1D(x, 128)
        x = Dropout(0.3)(x)
        x = Dense(classes_number, activation='softmax')(x)
        return x

    def build_model(self):
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

    def train(self, x_train, y_train, x_valid, y_valid, learning_rate=1e-3, epochs=100):
        self.model.compile(loss="categorical_crossentropy",
                           optimizer=Adam(learning_rate),
                           metrics=['accuracy'])

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
        start_time = time.time()
        predictions = self.model.predict(data)
        end_time = time.time() - start_time

        if verbose:
            print(f"Prediction time: {end_time} seconds.")

        return predictions

    def save(self, save_path: str):
        self.model.save_weights(save_path)
