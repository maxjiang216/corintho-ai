import numpy as np
from copy import deepcopy


class Node:
    """
    Node in Monte Carlo Search Tree
    """

    def __init__(self, game, depth, parent=None):
        self.game = game
        self.searches = 1
        self.evaluation = 0
        self.depth = depth
        self.parent = parent
        self.moves = None
        self.probabilities = None
        self.children = None
        self.visits = None
        self.noise = None

    def string(self, n):
        out = "(" + str(self.evaluation)
        if self.children is not None:
            for child in self.children:
                if child is not None:
                    out += "\n" + " " * n + child.string(n + 1)
        out += ")"
        return out


class TrainMC:
    """
    Monte Carlo Search Tree used for training

    """

    def __init__(self, root, iterations=200, c_puct=1, epsilon=0.25):
        self.root = root
        self.iterations = iterations
        self.c_puct = c_puct
        self.epsilon = epsilon
        self.rng = np.random.Generator(np.random.PCG64())
        self.iterations_done = 0
        self.cur_node = None

    def choose_move(self, evaluations=None):
        """ """
        # Result to request for evaluations
        if evaluations is not None:
            
        for i in range(self.iterations):
            self.search(self.root)
        move_choice = None
        # Choose with weighted random for the first 2 moves from each player (temperature = 1)
        if self.root.depth < 4:
            move_choice = self.rng.choice(
                len(self.root.moves), p=self.root.visits / sum(self.root.visits)
            )
        # Otherwise, choose randomly between the moves with the most visits/searches
        else:
            max_value = self.root.visits[0]
            move_choices = [0]
            for i in range(1, len(self.root.moves)):
                visits = 0
                if self.root.children[i] is not None:
                    visits = self.root.visits[i]
                if visits > max_value:
                    max_value = visits
                    move_choices = [i]
                elif visits == max_value:
                    move_choices.append(i)
            move_choice = self.rng.choice(move_choices)
        move = self.root.moves[move_choice]
        self.root = self.root.children[move_choice]
        return move

    def force_move(self, move_choice):
        """
        Force choose a move
        Used for opponent's moves
        """
        # Unexplored state
        if self.root.children[move_choice] is None:
            new_game = deepcopy(self.root.game)
            new_game.do_move(self.root.moves[move_choice])
            new_evaluation = self.evaluator.evaluate(new_game)
            self.root = Node(
                new_game,
                new_evaluation,
                self.root.depth + 1,
            )
        else:
            self.root = self.root.children[move_choice]

    def receive_evaluations(self, evaluations):
        '''
        array ->
        Update nodes with newly received evaluations
        '''
        self.cur_node.evaluation = evaluations[0]
        self.cur_node.probabilities = evaluations[1]
        # Propagate evaluation to parent nodes
        while self.cur_node.parent is not None:
            self.cur_node.parent.evaluation += self.cur_node.evaluation * -1
            self.cur_node = self.cur_node.parent

    def search(self, node):
        """
        Node ->
        Search a node
        """

        node.searches += 1

        # First time searching this node
        if node.moves is None:
            node.moves = []
            legal_moves = node.game.get_legal_moves()
            node.probabilties = np.ma.masked_array(node.probabilities, legal_moves)
            for i in range(96):
                if legal_moves[i] > 0:
                    node.moves.append(i)
            node.probabilities = self.move_guider.generate(node.game)
            node.children = [None] * len(node.moves)
            node.visits = np.full(len(node.moves), 0)
            node.noise = self.rng.dirichlet(np.full(len(node.moves), 0.3))

        # Not terminal node
        if len(node.moves) > 0:
            max_value = -1
            move_choice = -1
            for i in range(len(node.moves)):
                u = 0
                # If not visited, set action value to 0
                if node.children[i] is None:
                    u = self.c_puct * node.probabilities[i] * np.sqrt(node.searches - 1)
                else:
                    u = -1 * node.children[i].evaluation / node.children[i].searches + (
                        self.c_puct
                        * ((1 - self.epsilon) * node.probabilities[i] + node.noise[i])
                        * np.sqrt(node.searches - 1)
                        / (node.children[i].searches + 1)
                    )
                if u > max_value:
                    max_value = u
                    move_choice = i

            node.visits[move_choice] += 1

            # Exploring new node
            if node.children[move_choice] is None:
                new_game = deepcopy(node.game)
                new_game.do_move(node.moves[move_choice])
                node.children[move_choice] = Node(
                    new_game,
                    node.depth + 1,
                    node,
                )
                # Prepare to update the new node with received evaluations
                self.cur_node = node.children[move_choice]
                # Request evaluations
                return True
            else:
                self.search(node.children[move_choice])

        # Terminal node
        else:
            # Propagate evaluation to parent nodes
            while self.cur_node.parent is not None:
                self.cur_node.parent.evaluation += self.cur_node.evaluation * -1
                self.cur_node = self.cur_node.parent

        return -1 * new_evaluation
