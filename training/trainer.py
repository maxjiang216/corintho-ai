import numpy as np
import time
from selfplayer import SelfPlayer


def helper(game, evaluations):
    """Helper function for multiprocessing"""
    return game.play(evaluations)


class Trainer:
    """
    Trains with multiple games
    Collects training samples
    """

    def __init__(self, model, model2, batch_size=32, num_games=3000, test=False):
        """
        (int) -> Trainer
        Will train with num_games games concurrently
        """

        self.games = []
        self.model = model
        self.model2 = model2
        self.batch_size = batch_size
        self.test = test
        if test:
            for i in range(num_games):
                self.games.append(SelfPlayer(test=True, seed=i % 2))
        else:
            for _ in range(num_games):
                self.games.append(SelfPlayer())

    def train_generation(self):
        """
        Play all num_games games and get training samples
        """

        # Play games

        evaluations = [None] * len(self.games)
        evaluations_done = 0
        start_time = time.time()

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
                    x=np.array(positions), batch_size=len(self.games), verbose=0
                )
                evaluations = list(zip(res[0], res[1]))
            evaluations_done += 1
            if evaluations_done % 50 == 0:
                time_taken = time.time() - start_time
                if time_taken < 60:
                    print(
                        f"{evaluations_done} evaluations completed in {time.time()-start_time:.1f} seconds"
                    )
                else:
                    print(
                        f"{evaluations_done} evaluations completed in {(time.time()-start_time)/60:.1f} minutes"
                    )
                print(
                    f"Predicted time to complete: {26.67*200*time_taken/evaluations_done/60/60:.2f} hours\n{(26.67*200-evaluations_done)*time_taken/evaluations_done/60/60:.2f} hours left"
                )

        # Compile logs
        # Return game histories and outcome as string to be written into file
        game_logs = ""
        total_turns = 0
        for i, game in enumerate(self.games):
            game_logs += f"GAME {i}\nRESULT: {game.game.outcome}\n"
            game_logs += "\n".join(game.logs) + "\n\n"
            total_turns += len(game.logs)
        game_logs = (
            f"AVERAGE NUMBER OF TURNS: {total_turns / 2 / len(self.games):.2f}\n"
            + game_logs
        )

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
            self.model2.fit(
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
            return (self.model2.get_weights(), game_logs)

        # Score of first player (first model)
        score = 0
        for game in self.games:
            score += (game.game.outcome * (-1) ** game.seed + 1) / 2

        # If testing, return win rate
        return (score / len(self.games), game_logs)
