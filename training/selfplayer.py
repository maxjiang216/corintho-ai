from implement.game import Game
from mc.mcplayer import MonteCarloPlayer


class SelfPlayer:
    """
    Interface to do self play during training
    """

    def __init__(self, iterations=200):
        """
        (int) -> SelfPlayers
        """

        self.game = Game()
        self.players = [MonteCarloPlayer(iterations), MonteCarloPlayer(iterations)]
        self.samples = None

    def play(self, evaluations):
        """
        array ->
        Plays until the next evaluation is needed
        """
