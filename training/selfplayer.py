import numpy as np
import time
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

    def __init__(self, iterations=200):
        """
        (int) -> SelfPlayers
        """

        self.game = Game()
        self.players = [TrainMC(Game(), iterations), TrainMC(Game(), iterations)]
        self.samples = []
        self.probability_labels = []

    def play(self, evaluations=None):
        """
        array ->
        Plays until the next evaluation is needed
        """

        times = [0, 0, 0, 0, 0, 0, 0]

        t0 = time.time()

        # While the game is still going
        if self.game.outcome is None:
            t4 = time.time()
            res = self.players[self.game.to_play].choose_move(evaluations)
            times[5] += time.time() - t4
            for i in range(len(res[-1])):
                times[i] += res[-1][i]
            while res[0] == "move" and self.game.outcome is None:
                self.players[1 - self.game.to_play].receive_opp_move(
                    res[2],  # move choice
                    self.players[self.game.to_play].root.probabilities,
                )
                self.samples.append(self.game.get_vector())
                self.probability_labels.append(res[3])
                self.game.do_move(res[1])  # move
                if self.game.outcome is None:
                    res = self.players[self.game.to_play].choose_move()
                    for i in range(len(res[-1])):
                        times[i] += res[-1][i]
            if res[0] == "eval":  # eval
                # Propagate up
                times[-1] = time.time() - t0
                return (res[1], self, times)

        # If game is done
        times[-1] = time.time() - t0
        return (np.zeros(70), self, times)

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
