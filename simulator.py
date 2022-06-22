from game import Game


class Simulator:
    def __init__(self, player1, player2):
        self.players = [player1, player2]
        self.game = Game()

    def play_game(self):
        """
        Plays game with player1 and player2
        returns winner
        """
        while True:
            move = self.players[self.game.to_play].get_move(
                self.game.get_canonical(), self.game.get_legal_moves()
            )
            self.players[1 - self.game.to_play].receive_opp_move(move)
            result = self.game.do_move(move)
            # print(str(self.game))
            if result is not None:
                return result
