from game import Game


class Simulator:
    def __init__(self, player1, player2):
        self.players = [player1, player2]

    def play_game(self, train=False):
        """
        (bool) -> bool
        Plays game with player1 and player2
        If train is True, adds training samples to self.samples
        returns winner
        """
        game = Game()
        if train:
            samples = []
        while True:
            if train:
                samples.append(game.get_vector())
            legal_moves = game.get_legal_moves()
            out = ""
            for move in legal_moves:
                out += str(move) + " "
            print(out)
            print(game.board.lines)
            move = self.players[game.to_play].get_move(game, legal_moves)
            self.players[1 - game.to_play].receive_opp_move(move)
            result = game.do_move(move)
            if result is not None:
                if train:
                    return (
                        [sample for sample in samples],
                        [result * (-1) ** i for i in range(len(samples))],
                    )
                return result
