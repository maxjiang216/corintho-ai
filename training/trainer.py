import numpy as np
from multiprocessing import Pool, cpu_count
from training.selfplayer import SelfPlayer


class Trainer:
    """
    Trains with multiple games concurrently
    Collects training samples
    """

    def __init__(self, evaluator, move_guider, batch_size=64, num_games=2000):
        """
        (int) -> Trainer
        Will train with num_games games concurrently
        """

        self.games = []
        self.evaluator = evaluator
        self.move_guider = move_guider
        self.batch_size = batch_size
        for _ in range(num_games):
            self.games.append(SelfPlayer())

    def train_generation(self):
        """
        Play all num_games games and get training samples
        """

        def helper(game, evaluations):
            return game.play(evaluations)

        # Play games
        with Pool(processes=cpu_count()) as pool:

            evaluations = [None] * len(self.games)
            probabilities = [None] * len(self.games)

            while True:
                positions = pool.starmap(
                    helper, zip(self.games, zip(evaluations, probabilities))
                )
                games_completed = 0
                for position in positions:
                    if not np.any(position):
                        games_completed += 1
                if games_completed >= len(self.games):
                    break
                evaluations = self.evaluator.predict(x=positions)
                probabilities = self.move_guider.predict(x=positions)

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
        self.evaluator.fit(
            x=np.array(samples),
            y=np.array(evaluation_labels),
            batch_size=self.batch_size,
            epochs=1,
            shuffle=True,
        )
        self.move_guider.fit(
            x=np.array(samples),
            y=np.array(probability_labels),
            batch_size=self.batch_size,
            epochs=1,
            shuffle=True,
        )

        # Return weights
        return self.evaluator.get_weights(), self.move_guider.get_weights()
