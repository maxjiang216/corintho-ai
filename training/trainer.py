import numpy as np
from multiprocessing import Pool, cpu_count
from selfplayer import SelfPlayer


def helper(game, evaluations):
    """Helper function"""
    return game.play(evaluations)


class Trainer:
    """
    Trains with multiple games concurrently
    Collects training samples
    """

    def __init__(self, model, batch_size=32, num_games=2):
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

            while True:
                res = pool.starmap(helper, zip(self.games, evaluations))
                positions = []
                for i, item in enumerate(res):
                    self.games[i] = item[1]
                    positions.append(item[0])
                games_completed = 0
                for game in self.games:
                    if game.game.outcome is not None:
                        games_completed += 1
                if games_completed >= len(self.games):
                    break
                res = self.model.predict(x=np.array(positions))
                evaluations = zip(res[0], res[1])

            pool.close()

        # Collect samples
        samples = []
        evaluation_labels = []
        probability_labels = []
        for game in self.games:
            (
                cur_samples,
                cur_evaluation_labels,
                cur_probability_labels,
            ) = game.get_samples()
            samples.extend(cur_samples)
            evaluation_labels.extend(cur_evaluation_labels)
            probability_labels.extend(cur_probability_labels)

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
