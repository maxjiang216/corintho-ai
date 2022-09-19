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

    def __init__(
        self,
        iterations=200,
        testing=False,
        seed=0,
        samples_file=None,
        logging=False,
        logs_file=None,
    ):
        """
        (int) -> SelfPlayers
        """

        self.game = Game()
        self.testing = testing
        self.seed = seed
        self.samples_file = samples_file
        self.logging = logging
        self.logs_file = logs_file
        if testing:
            self.players = [
                TrainMC(
                    Game(),
                    iterations,
                    player_num=seed % 2,
                    testing=True,
                    logging=logging,
                ),
                TrainMC(
                    Game(),
                    iterations,
                    player_num=(seed + 1) % 2,
                    testing=True,
                    logging=logging,
                ),
            ]
        else:
            self.players = [
                TrainMC(
                    Game(),
                    iterations,
                    testing=False,
                    logging=logging,
                ),
                TrainMC(
                    Game(),
                    iterations,
                    testing=False,
                    logging=logging,
                ),
            ]
        self.samples = []
        self.probability_labels = []

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
                )

                if not self.testing:
                    self.samples_file.write(
                        f"{self.game.get_vector()}\t{res[3]}\t{self.seed}\n"
                    )

                self.game.do_move(res[1])  # do move

                if self.logging:  # write to log if needed
                    self.logs_file.write(f"{res[-1]}\n{str(self.game)}\n\n")

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
        # Draw, all labels are 0
        if self.game.outcome == 0:
            evaluation_labels = [0] * len(self.samples)
        else:
            evaluation_labels = [
                (-1) ** (len(self.samples) - i - 1) for i in range(len(self.samples))
            ]

        return self.samples, evaluation_labels, self.probability_labels
