import numpy as np
from multiprocessing import Pool, cpu_count
import time
from selfplayer import SelfPlayer


class Trainer:
    """
    Trains with multiple games concurrently
    Collects training samples
    """

    def __init__(self, model, batch_size=32, num_games=1):
        """
        (int) -> Trainer
        Will train with num_games games concurrently
        """

        self.games = []
        self.model = model
        self.batch_size = batch_size
        for _ in range(num_games):
            self.games.append(SelfPlayer())

    def play_games(self, i, evaluations):
        """Helper function for multiprocessing"""
        return self.games[i].play(evaluations)

    def train_generation(self):
        """
        Play all num_games games and get training samples
        """

        # Play games
        with Pool(processes=cpu_count()) as pool:

            evaluations = [None] * len(self.games)

            start_time = time.time()
            searches = 0

            # Training samples
            samples = []
            evaluation_labels = []
            probability_labels = []

            while True:

                t2 = time.time()
                res = pool.starmap(
                    self.play_games, zip(np.arange(len(self.games)), evaluations)
                )
                t3 = time.time() - t2
                searches += 1
                print("trainer")
                print((time.time() - start_time) / searches)
                print(t3)

                games = []
                positions = []
                t4 = time.time()
                t6 = res[0][2]
                for i, item in enumerate(res):
                    if item[1].game.outcome is None:
                        games.append(self.games[i])
                    else:
                        (
                            cur_samples,
                            cur_evaluation_labels,
                            cur_probability_labels,
                        ) = item[1].get_samples()
                        samples.extend(cur_samples)
                        evaluation_labels.extend(cur_evaluation_labels)
                        probability_labels.extend(cur_probability_labels)
                    positions.append(item[0])
                self.games = games
                t5 = time.time() - t4
                print(t5)
                print(t6)
                # Done all games
                if len(self.games) == 0:
                    print(time.time() - start_time)
                    print("DONE")
                    break
                res = self.model.predict(x=np.array(positions), verbose=0)
                evaluations = zip(res[0], res[1])

            pool.close()

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
