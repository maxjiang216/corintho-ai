import keras.api._v2.keras as keras
from keras.api._v2.keras import Input, regularizers
from keras.api._v2.keras.models import Model
from keras.api._v2.keras.layers import Activation, Dense, BatchNormalization
from keras.api._v2.keras.optimizers import Adam
from neuralnet import NeuralNet
from simulator import Simulator
from mcplayer import MonteCarloPlayer
from player import RandomPlayer

# Seems to be important guard to avoid some kind of infinite recursion
# Would be useful to add command line variables
if __name__ == "__main__":

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
    eval_layer = Dense(units=1, activation="tanh")(layer_4)
    # Add this later
    # guide_layer = Dense(units=96, activation="softmax")(layer_4)
    # Add to this line, too
    # Maybe we should split this into two models, since they are called at different times.
    model = Model(inputs=input_layer, outputs=eval_layer)
    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss=keras.losses.MeanSquaredError(),
    )
    simulator = Simulator(RandomPlayer(), RandomPlayer())
    neural_net = NeuralNet(model, simulator, batch_size=10)
    neural_net.train_generation(500, 8)
    neural_net.check()
    print(neural_net.model.summary())
