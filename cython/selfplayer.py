import numpy as np
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from game import Game
from trainmc import TrainMC


class SelfPlayer:
    """
    Interface to do self play during training
    """

    def __init__(
        self, iterations=200, series_length=1, testing=False, logging=False, seed=0
    ):
        """
        (int) -> SelfPlayers
        """

        self.game = Game()
        self.iterations = iterations
        self.testing = testing
        if testing:
            self.seed = seed
            self.players = [
                TrainMC(Game(), iterations, player_num=seed % 2, testing=True),
                TrainMC(Game(), iterations, player_num=(seed + 1) % 2, testing=True),
            ]
        else:
            self.players = [TrainMC(Game(), iterations), TrainMC(Game(), iterations)]
        self.samples = []
        self.probability_labels = []
        for _ in range(series_length):
            self.samples.append([])
            self.probability_labels.append([])
        self.logs = []
        self.logging = logging
        self.series_length = series_length
        self.games_played = 0
        self.game_outcomes = []

    def play(self, evaluations=None):
        """
        array ->
        Plays until the next evaluation is needed
        """

        # Game is done, possibly start new game in series
        if self.game.outcome is not None:
            evaluations = None
            if self.games_played + 1 < self.series_length:
                self.games_played += 1
                self.game = Game()
                if self.testing:
                    self.players = [
                        TrainMC(
                            Game(),
                            self.iterations,
                            player_num=self.seed % 2,
                            testing=True,
                        ),
                        TrainMC(
                            Game(),
                            self.iterations,
                            player_num=(self.seed + 1) % 2,
                            testing=True,
                        ),
                    ]
                else:
                    self.players = [
                        TrainMC(Game(), self.iterations),
                        TrainMC(Game(), self.iterations),
                    ]
            else:
                # If game is done
                return np.zeros(70)

        # While the game is still going
        res = self.players[self.game.to_play].choose_move(evaluations)
        while res[0] == "move" and self.game.outcome is None:
            self.players[1 - self.game.to_play].receive_opp_move(
                res[2],  # move choice
            )
            self.samples[self.games_played].append(self.game.get_vector())
            self.probability_labels[self.games_played].append(res[3])
            self.game.do_move(res[1])  # move
            if self.logging:
                self.logs.append(str(res[-1]))
                self.logs.append(str(self.game))  # Keep game logs
            if self.game.outcome is None:
                res = self.players[self.game.to_play].choose_move()
        # First time game is done, record result
        if self.game.outcome is not None:
            self.game_outcomes.append(self.game.outcome)
            # Indicate we are not done. Is there a better way to do this?
            return np.ones(70)
        # We have eval, propagate up
        return res[1]

    def get_samples(self):
        """
        -> array
        Get array of training samples
        Game positions tagged with final game result
        Returns empty array if game is not complete
        """

        # Not done playing
        if self.games_played + 1 < self.series_length:
            return [], [], []
        evaluation_labels = []
        samples = []
        probability_labels = []
        for i in range(self.series_length):
            samples.extend(self.samples[i])
            probability_labels.extend(self.probability_labels[i])
            # Draw, all labels are 0
            if self.game_outcomes[i] == 0:
                evaluation_labels.extend([0] * len(self.samples[i]))
            else:
                evaluation_labels.extend(
                    [
                        (-1) ** (len(self.samples[i]) - j - 1)
                        for j in range(len(self.samples[i]))
                    ]
                )

        return samples, evaluation_labels, probability_labels
