from mc.montecarlo import MonteCarlo
from implement.simulator import Simulator
from implement.player import RandomPlayer, HumanPlayer
from mc.mcplayer import MonteCarloPlayer
from implement.game import Game
from neural.neuralnet import NeuralNet
import multiprocessing as mp

simulator = Simulator(MonteCarloPlayer(60), MonteCarloPlayer(60))
simulator.play_game()

if __name__ == "__main__":
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import Adam

    print("Number of processors: ", mp.cpu_count())

    model = Sequential(
        [
            Dense(units=70, input_shape=(70,), activation="relu"),
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
    print("COMPILED MODEL")
    simulator = Simulator(MonteCarloPlayer(60), MonteCarloPlayer(60))
    nn = NeuralNet(model, simulator, batch_size=10)
    k = 2
    print("DONE SETUP")
    nn.train_generation(num_games=k * 2, concurrency=k)
    print(nn.predict(Game()))
    nn.check()
