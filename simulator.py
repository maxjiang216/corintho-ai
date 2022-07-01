from game import Game


class Simulator:
    def __init__(self, player1, player2):
        self.players = [player1, player2]
        self.game = Game()
        self.samples = []

    def play_game(self, train=False):
        """
        (bool) -> bool
        Plays game with player1 and player2
        If train is True, adds training samples to self.samples
        returns winner
        """
        if train:
            samples = []
        while True:
            samples.append(self.game.get_vector())
            move = self.players[self.game.to_play].get_move(
                self.game, self.game.get_legal_moves()
            )
            self.players[1 - self.game.to_play].receive_opp_move(move)
            result = self.game.do_move(move)
            # print(str(self.game))
            if result is not None:
                self.samples.extend(
                    [(sample, result * (-1) ** i) for i, sample in enumerate(samples)]
                )
                return result
