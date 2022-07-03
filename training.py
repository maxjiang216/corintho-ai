import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from neuralnet import NeuralNet
from simulator import Simulator
from mcplayer import MonteCarloPlayer
from player import RandomPlayer

# Seems to be important guard to avoid some kind of infinite recursion
# Would be useful to add command line variables
if __name__ == "__main__":

    model = Sequential(
        [
            # First layer need input_shape 70 for the vector representation of the game state
            Dense(units=70, input_shape=(70,), activation="relu"),
            # Neural net shape is somewhat arbitrary at this point
            # With the number of games tested so far, it seems to simply memorize positions
            Dense(units=80, activation="relu"),
            Dense(units=90, activation="relu"),
            Dense(units=100, activation="relu"),
            Dense(units=110, activation="relu"),
            Dense(units=120, activation="relu"),
            Dense(units=130, activation="relu"),
            Dense(units=140, activation="relu"),
            Dense(units=130, activation="relu"),
            Dense(units=120, activation="relu"),
            Dense(units=110, activation="relu"),
            Dense(units=100, activation="relu"),
            Dense(units=90, activation="relu"),
            Dense(units=80, activation="relu"),
            Dense(units=70, activation="relu"),
            Dense(units=1, activation="tanh"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=keras.losses.MeanSquaredError(),
    )
    simulator = Simulator(RandomPlayer(), RandomPlayer())
    neural_net = NeuralNet(model, simulator, batch_size=10)
    neural_net.train_generation(1000, 4)
    neural_net.check()
