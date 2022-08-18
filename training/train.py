import os
import pstats

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import keras.api._v2.keras as keras
from keras.api._v2.keras import Input, regularizers
from keras.api._v2.keras.models import Model
from keras.api._v2.keras.layers import Activation, Dense, BatchNormalization
from keras.api._v2.keras.optimizers import Adam
from trainer import Trainer

if __name__ == "__main__":

    import cProfile
    import pstats

    with cProfile.Profile() as pr:

        input_layer = Input(shape=(70,))
        layer_1 = Activation("relu")(
            BatchNormalization()(
                Dense(
                    units=100,
                    kernel_regularizer=regularizers.L2(1e-4),
                )(input_layer)
            )
        )
        layer_2 = Activation("relu")(
            BatchNormalization()(
                Dense(
                    units=100,
                    kernel_regularizer=regularizers.L2(1e-4),
                )(layer_1)
            )
        )
        layer_3 = Activation("relu")(
            BatchNormalization()(
                Dense(
                    units=100,
                    kernel_regularizer=regularizers.L2(1e-4),
                )(layer_2)
            )
        )
        layer_4 = Activation("relu")(
            BatchNormalization()(
                Dense(
                    units=100,
                    kernel_regularizer=regularizers.L2(1e-4),
                )(layer_3)
            )
        )
        eval_layer = Activation("relu")(
            BatchNormalization()(
                Dense(
                    units=100,
                    kernel_regularizer=regularizers.L2(1e-4),
                )(layer_4)
            )
        )
        eval_output = Dense(units=1, activation="tanh")(eval_layer)
        prob_layer = Activation("relu")(
            BatchNormalization()(
                Dense(
                    units=100,
                    kernel_regularizer=regularizers.L2(1e-4),
                )(layer_4)
            )
        )
        prob_output = Dense(units=96, activation="softmax")(prob_layer)

        model = Model(inputs=input_layer, outputs=[eval_output, prob_output])
        model.compile(
            optimizer=Adam(learning_rate=0.01),
            loss=[
                keras.losses.MeanSquaredError(),
                keras.losses.CategoricalCrossentropy(),
            ],
        )

        trainer = Trainer(
            model,
        )

        trainer.train_generation()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
