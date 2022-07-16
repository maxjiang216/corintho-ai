from implement.player import Player
from implement.move import Move
from mc.montecarlo import MonteCarlo, Node
from infra.evaluator import RandomEvaluator
from infra.moveguider import UniformGuider


class MonteCarloPlayer(Player):
    """
    Player using Monte Carlo Search Tree with used defined evaluator
    """

    def __init__(self, iterations=600):
        self.tree = None
        self.evaluator = RandomEvaluator()
        self.move_guider = UniformGuider()
        self.iterations = iterations

    def get_move(self, game, legal_moves):
        # first query or new game
        if self.tree is None:
            self.tree = MonteCarlo(
                Node(game, self.evaluator.evaluate(game)),
                self.evaluator,
                self.move_guider,
                iterations=self.iterations,
            )
        return self.tree.choose_move()

    def receive_opp_move(self, move):
        """
        Move tree down
        """
        if (
            self.tree is not None
            and self.tree.root.moves is not None
            and move in self.tree.root.moves
        ):
            self.tree.force_move(self.tree.root.moves.index(move))
