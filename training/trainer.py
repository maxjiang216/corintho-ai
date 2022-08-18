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

                res = pool.starmap(helper, zip(self.games, evaluations))

                searches += 1
                print("trainer")
                print(len(self.games))
                print((time.time() - start_time) / searches)

                self.games = []
                positions = []
                for item in res:
                    print("trainer 60")
                    print(item)
                    if item[1].game.outcome is None:
                        self.games.append(item[1])
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
                print("trainer 72")
                print(len(self.games))
                # Done all games
                if len(self.games) == 0:
                    print("DONE")
                    break
                print(positions)
                res = self.model.predict(x=np.array(positions), verbose=0)
                evaluations = zip(res[0], res[1])

            pool.close()

        print(samples)

        # Train neural nets
        self.model.fit(
            x=np.array(samples),
            y=[np.array(evaluation_labels), np.array(probability_labels)],
            batch_size=self.batch_size,
            epochs=1,
            shuffle=True,
        )

        # Return weights
        return self.model.get_weights()
