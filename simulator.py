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
        to_play = 0
        while True:
            move = self.players[to_play].get_move(self.game)
            result = self.game.do_move(move)
            if result is not None:
                return result
            to_play = 1 - to_play
            # flip board perspective
