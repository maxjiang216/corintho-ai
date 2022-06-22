import random
import numpy as np


class Node:
    """
    Node in Monte Carlo Search Tree
    """

    def __init__(self, game, evaluation):
        self.game = game
        self.searches = 1
        self.evaluation = evaluation
        self.moves = None
        self.probabilities = None
        self.children = None
        self.visits = None


class MonteCarlo:
    """
    Monte Carlo Search Tree with user-defined position evaulator
    """

    def __init__(self, root, evaluator, move_guider, c_puct=1):
        self.root = root
        self.evaluator = evaluator
        self.move_guider = move_guider
        self.c_puct = c_puct

    def search(self, node):
        """
        Search a node
        """
        if node.moves is None:
            node.moves = node.game.get_legal_moves()
            node.probabilities = self.move_guider.generate(node.game)
            node.children = [None] * len(node.moves)
            node.visits = np.full(len(node.moves), 0)

        # Result of search
        new_evaluation = 0
        # Not terminal node
        if len(node.moves) > 0:
            max_value = 0
            move_choice = -1
            for i in range(len(node.moves)):
                u = 0
                # If not visited, set action value to 0
                if node.children[i] is None:
                    u = self.c_puct * node.probabilities[i] * np.sqrt(node.searches - 1)
                else:
                    u = node.children[i].evaluation / node.children[i].searches + (
                        self.c_puct
                        * node.probabilities[i]
                        * np.sqrt(node.searches - 1)
                        / node.children[i].searches
                    )
                if u > max_value:
                    max_value = u
                    move_choice = i

            # Exploring new node
            if node.children[move_choice] is None:
                new_game = node.game.do_move(node.moves[move_choice]).get_canonical()
                new_evaluation = self.evaluator(new_game)
                node.children[move_choice] = Node(
                    new_game,
                    new_evaluation,
                )
            else:
                new_evaluation = self.search(node.children[move_choice])

            node.evauation += new_evaluation
            node.searches += 1
            node.visits[move_choice] += 1
        # Terminal node
        else:
            new_evaluation = node.evaluation

        return new_evaluation
