import numpy as np
from multiprocessing import Pool, cpu_count
from selfplayer import SelfPlayer


def helper(game, evaluations):
    """Helper function for multiprocessing"""
    return game.play(evaluations)


class Trainer:
    """
    Trains with multiple games
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
        if model2 is not None:
            self.test = True
            self.model2 = model2
            for i in range(num_games):
                self.games.append(SelfPlayer(test=True, seed=i % 2))
        else:
            self.test = False
            for _ in range(num_games):
                self.games.append(SelfPlayer())

    def train_generation(self):
        """
        Play all num_games games and get training samples
        """

        # Play games

        evaluations = [None] * len(self.games)

        while True:

            res = []
            for i, game in enumerate(self.games):
                res.append(game.play(evaluations[i]))

            positions = []
            games_done = 0
            for item in res:
                # Game is done
                if max(item) == 0:
                    games_done += 1
                positions.append(item)
            # Done all games
            if games_done == len(self.games):
                break
            if self.test:
                res1 = self.model.predict(
                    x=np.array(positions), batch_size=self.batch_size, verbose=0
                )
                res2 = self.model2.predict(
                    x=np.array(positions), batch_size=self.batch_size, verbose=0
                )
                evaluations = list(
                    zip(list(zip(res1[0], res1[1])), list(zip(res2[0], res2[1])))
                )
            else:
                res = self.model.predict(
                    x=np.array(positions), batch_size=self.batch_size, verbose=0
                )
                evaluations = list(zip(res[0], res[1]))

        if not self.test:

            # Training samples
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

        # Score of first player (first model)
        score = 0
        for game in self.games:
            score += (game.game.outcome * (-1) ** game.seed + 1) / 2

        # If testing, return win rate
        return score / len(self.games)
