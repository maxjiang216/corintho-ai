import numpy as np
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
        self.players = [TrainMC(iterations), TrainMC(iterations)]
        self.samples = []
        self.probability_labels = []

    def play(self, evaluations=None):
        """
        array ->
        Plays until the next evaluation is needed
        """

        # While the game is still going
        if self.game.outcome is not None:
            res = self.players[self.game.to_play].choose_move(evaluations)

            if res[0] == "move":
                self.players[1 - self.game.to_play].receive_opp_move(
                    res[2],  # move choice
                    self.players[self.game.to_play].root.probabilities,
                )
                self.samples.append(self.game.get_vector())
                self.probability_labels.append(res[2])
                self.game.do_move(res[1])  # move
                # Game not finished
                if self.game.outcome is None:
                    # Do next move (there will be at least one search)
                    return self.players[self.game.to_play].choose_move()
            else:  # eval
                # Propagate up
                return res

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
            return [], [], [], []
        evaluation_labels = []
        for i, sample in enumerate[self.samples]:
            evaluation_labels.append(
                self.game.outcome * (-1) ** (len(self.samples) - i - 1),
            )

        return self.samples, evaluation_labels, probability_labels
