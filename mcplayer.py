from player import Player
from montecarlo import MonteCarlo, Node
from evaluator import RandomEvaluator
from moveguider import UniformGuider


class MonteCarloPlayer(Player):
    """
    Player using Monte Carlo Search Tree with used defined evaluator
    """

    def __init__(self):
        self.tree = None
        self.evaluator = RandomEvaluator()
        self.move_guider = UniformGuider()

    def get_move(self, game, legal_moves):
        # first query or new game
        if self.tree is None or self.tree.root.moves != legal_moves:
            self.tree = MonteCarlo(
                Node(game, self.evaluator.evaluate(game)),
                self.evaluator,
                self.move_guider,
            )
        return self.tree.choose_move()

    def receive_opp_move(self, move):
        """
        Move tree down
        """
        if self.tree is not None and move in self.tree.root.moves:
            self.tree.force_move(self.tree.root.moves.index(move))
