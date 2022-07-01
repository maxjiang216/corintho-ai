import numpy as np
import time
import multiprocessing as mp
from simulator import Simulator
from player import RandomPlayer
from mcplayer import MonteCarloPlayer
from game import Game
import multiprocessing as mp
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class NeuralNet:
    """
    Neural Net for evaluating Corintho positions
    """

    def __init__(self, model, simulator, batch_size=16):
        self.model = model
        self.simulator = simulator
        self.batch_size = batch_size
        self.samples = []
        self.labels = []

    @staticmethod
    def play_games(args):
        """ """
        simulator = args[0]
        num_games = args[1]
        counter = args[2]
        samples = []
        labels = []
        for i in range(num_games):
            temp_samples, temp_labels = simulator.play_game(train=True)
            samples.extend(temp_samples)
            labels.extend(temp_labels)
            print("Done game {0}.{1}".format(counter, i))
        print("Done batch {0}".format(counter))
        return (samples, labels)

    def train_generation(self, num_games=100, concurrency=10):
        """
        int ->
        Generating training samples and fits model
        Players num_games games to generate samples
        """

        args = [
            (self.simulator, num_games // concurrency, i) for i in range(concurrency)
        ]
        with mp.Pool(concurrency) as pool:
            t = time.time()
            result = pool.map(self.play_games, args)
            print((time.time() - t) / ((num_games // concurrency) * concurrency))
            pool.close()
            for game in result:
                self.samples.extend(game[0])
                self.labels.extend(game[1])

        self.model.fit(
            x=np.array(self.samples),
            y=np.array(self.labels),
            batch_size=self.batch_size,
            epochs=3,
            shuffle=True,
        )

    def predict(self, game):
        """
        Game -> float
        Gives evaluation of game
        """
        return self.model.predict(x=np.array([game.get_vector()]))

    def check(self):
        f = open("./models/check.txt", "w")
        predictions = self.model.predict(x=np.array(self.samples))
        for i in range(len(self.samples)):
            f.write(
                "{0} {1} {2}\n".format(predictions[i], self.labels[i], self.samples[i])
            )

    def f(self):
        print(1)
        return 1


def fun(net, ngames, c):
    return net.play_games(ngames, c)


if __name__ == "__main__":

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
    simulator = Simulator(MonteCarloPlayer(60), MonteCarloPlayer(60))
    neural_net = NeuralNet(model, simulator, batch_size=10)
    neural_net.train_generation(256, 8)
