from implement.game import Game
from implement.move import Move


class Simulator:
    def __init__(self, player1, player2):
        self.players = [player1, player2]

    def play_game(self):
        """
        (bool) -> bool
        Plays game with player1 and player2
        returns winner
        """
        game = Game()
        while True:
            legal_moves = game.get_legal_moves()
            print("SIMULATOR")
            print(legal_moves)
            print(str(game))
            move = self.players[game.to_play].get_move(game, legal_moves)
            self.players[1 - game.to_play].receive_opp_move(move)
            result = game.do_move(move)
            if result is not None:
                return result
