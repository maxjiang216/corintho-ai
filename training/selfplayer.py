import numpy as np
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from implement.game import Game
from training.trainmc import TrainMC


class SelfPlayer:
    """
    Interface to do self play during training
    """

    def __init__(self, iterations=200, test=False, seed=0):
        """
        (int) -> SelfPlayers
        """

        self.game = Game()
        if test:
            self.seed = seed
            self.players = [
                TrainMC(Game(), iterations, player_num=seed % 2),
                TrainMC(Game(), iterations, player_num=(seed + 1) % 2),
            ]
        else:
            self.players = [TrainMC(Game(), iterations), TrainMC(Game(), iterations)]
        self.samples = []
        self.probability_labels = []
        self.logs = []

    def play(self, evaluations=None):
        """
        array ->
        Plays until the next evaluation is needed
        """

        # While the game is still going
        if self.game.outcome is None:
            res = self.players[self.game.to_play].choose_move(evaluations)
            while res[0] == "move" and self.game.outcome is None:
                self.players[1 - self.game.to_play].receive_opp_move(
                    res[2],  # move choice
                    self.players[self.game.to_play].root.probabilities,
                )
                self.samples.append(self.game.get_vector())
                self.probability_labels.append(res[3])
                self.game.do_move(res[1])  # move
                self.logs.append(str(self.game))  # Keep game logs
                if self.game.outcome is None:
                    res = self.players[self.game.to_play].choose_move()
            if res[0] == "eval":  # eval
                # Propagate up
                return res[1]

        # If game is done
        return np.zeros(70)

    def get_samples(self):
        """
        -> array
        Get array of training samples
        Game positions tagged with final game result
        Returns empty array if game is not complete
        """

        if self.game.outcome is None:
            return [], [], []
        evaluation_labels = []
        for i, _ in enumerate(self.samples):
            evaluation_labels.append(
                self.game.outcome * (-1) ** (len(self.samples) - i - 1),
            )

        return self.samples, evaluation_labels, self.probability_labels
