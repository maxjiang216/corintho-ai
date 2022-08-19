import numpy as np
from multiprocessing import Pool, cpu_count
import time
from selfplayer import SelfPlayer


def helper(game, evaluations):
    """Helper function for multiprocessing"""
    return game.play(evaluations)


class Trainer:
    """
    Trains with multiple games concurrently
    Collects training samples
    """

    def __init__(self, model, batch_size=32, num_games=3000, model2=None):
        """
        (int) -> Trainer
        Will train with num_games games concurrently
        """

        self.games = []
        self.model = model
        self.batch_size = batch_size
        self.test = False
        if model2 is not None:
            self.test = True
            self.model2 = model2
        for _ in range(num_games):
            self.games.append(SelfPlayer())

    def train_generation(self):
        """
        Play all num_games games and get training samples
        """

        # Play games

        evaluations = [None] * len(self.games)

        start_time = time.time()
        searches = 0

        # Training samples
        samples = []
        evaluation_labels = []
        probability_labels = []

        while True:

            t2 = time.time()
            res = []
            for i, game in enumerate(self.games):
                res.append(game.play(evaluations[i]))
            t3 = time.time() - t2

            searches += 1
            print("trainer")
            print((time.time() - start_time) / searches)
            print(t3)

            self.games = []
            positions = []
            t4 = time.time()
            times = [0, 0, 0, 0, 0, 0, 0]
            for item in res:
                for i in range(len(item[-1])):
                    times[i] += item[-1][i]
                if item[1].game.outcome is None:
                    self.games.append(item[1])
                elif not self.test:
                    (
                        cur_samples,
                        cur_evaluation_labels,
                        cur_probability_labels,
                    ) = item[1].get_samples()
                    samples.extend(cur_samples)
                    evaluation_labels.extend(cur_evaluation_labels)
                    probability_labels.extend(cur_probability_labels)

                positions.append(item[0])
            t5 = time.time() - t4
            print(t5)
            print(times)
            # Done all games
            if len(self.games) == 0:
                print(time.time() - start_time)
                print("DONE")
                break
            res = self.model.predict(x=np.array(positions), batch_size=self.batch_size)
            evaluations = list(zip(res[0], res[1]))

        if not self.test:
            # Train neural nets
            self.model.fit(
                x=np.array(samples),
                y=[
                    np.array(evaluation_labels),
                    np.array(probability_labels),
                ],
                batch_size=self.batch_size,
                epochs=1,
                shuffle=True,
            )
            # Return weights
            return self.model.get_weights()

        # If testing, return win rate
        return
