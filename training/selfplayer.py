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
        self.samples = None

    def play(self, evaluations=None):
        """
        array ->
        Plays until the next evaluation is needed
        """

        if self.game.outcome is not None:
            res = self.players[self.game.to_play].choose_move(evaluations)

            if res[0] == "move":
                self.players[1 - self.game.to_play].receive_opp_move(
                    res[2],  # move choice
                    self.players[self.game.to_play].root.probabilities,
                )
                self.game.do_move(res[1])  # move
                # Game not finished
                if self.game.outcome is None:
                    # Do next move
                    return self.players[self.game.to_play].choose_move()
            else:  # eval
                # Propagate up
                return res

        # If game is done
        return None
