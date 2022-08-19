import numpy as np
from copy import deepcopy
import time


class Node:
    """
    Node in Monte Carlo Search Tree
    """

    def __init__(self, game, depth, parent=None, probabilities=None):
        self.game = game
        self.searches = 1
        self.evaluation = 0
        self.depth = depth
        self.parent = parent
        self.moves = None
        self.legal_moves = None
        self.probabilities = probabilities
        self.children = None
        self.visits = None
        self.noise = None


class TrainMC:
    """
    Monte Carlo Search Tree used for training

    """

    def __init__(self, game, iterations, c_puct=1, epsilon=0.25, player_num=None):
        self.root = Node(game, 0)
        self.iterations = iterations
        self.iterations_done = 0
        self.c_puct = c_puct
        self.epsilon = epsilon
        self.rng = np.random.Generator(np.random.PCG64())
        self.iterations_done = 0
        self.cur_node = None
        self.player_num = player_num

    def choose_move(self, evaluations=None):
        """
        Runs one search or chooses a move
        If search is run, ("eval", bool) is returned
        where bool indicates whether an evaluation is requested
        Otherwise, ("move", move) is returns, where move is the chosen move
        """

        times = [0, 0, 0, 0, 0]

        # Result to request for evaluations
        if evaluations is not None:
            # Choose evaluation from correct neural net
            if self.player_num is not None:
                self.receive_evaluations(evaluations[self.player_num])
            else:
                self.receive_evaluations(evaluations)

        # If we have not done enough searches, search again
        if self.iterations_done < self.iterations:
            t1 = time.time()
            res = self.search(self.root)
            times[4] += time.time() - t1
            for i in range(len(res[-1])):
                times[i] += res[-1][i]
            # Search until an evaluation is needed
            while not res[0] and self.iterations_done < self.iterations:
                t1 = time.time()
                res = self.search(self.root)
                times[4] += time.time() - t1
                for i in range(len(res[-1])):
                    times[i] += res[-1][i]
            if res[0]:
                return ("eval", res[1], times)
        # Otherwise, choose a move
        self.iterations_done = 0
        move_choice = None
        # Choose with weighted random for the first 2 moves from each player (temperature = 1)
        if self.root.depth < 4:
            move_choice = self.rng.choice(
                len(self.root.moves), p=self.root.visits / sum(self.root.visits)
            )
        # Otherwise, choose randomly between the moves with the most visits/searches
        else:
            max_value = -2
            move_choice = 0
            # Start on random index to ensure random selection in case of ties
            start = self.rng.integers(0, len(self.root.moves))
            for i in range(len(self.root.moves)):
                id = (i + start) % len(self.root.moves)
                visits = 0
                if self.root.children[id] is not None:
                    visits = self.root.visits[id]
                if visits > max_value:
                    max_value = visits
                    move_choice = id
        final_ratio = np.zeros(96)
        total_visits = sum(
            self.root.visits
        )  # Might be different from self.iterations due to reusing tree
        for i, move in enumerate(self.root.moves):
            if self.root.visits[i] > 0:
                final_ratio[move] = self.root.visits[i] / total_visits
        move = self.root.moves[move_choice]
        self.root = self.root.children[move_choice]
        return ("move", move, move_choice, final_ratio, times)

    def receive_opp_move(self, move_choice, probabilities=None):
        """
        Force choose a move
        Used for opponent's moves
        """

        # Receiving first move
        # May be used if iterations is lower than legal move number
        if self.root.children is None:
            self.root.moves = []
            for i, is_legal in enumerate(self.root.game.get_legal_moves()):
                if is_legal == 1:
                    self.root.moves.append(i)
            # Cannot be game end
            res = self.root.game.do_move(self.root.moves[move_choice])
            self.root = Node(
                self.root.game,
                self.root.depth + 1,
                probabilities=probabilities,
            )
            self.root.legal_moves = res[1]
        # Unexplored state
        elif self.root.children[move_choice] is None:
            res = self.root.game.do_move(self.root.moves[move_choice])
            self.root = Node(
                self.root.game,
                self.root.depth + 1,
                probabilities=probabilities,
            )
            # Not a terminal state
            if res[0] is None:
                self.root.legal_moves = res[1]
        else:
            self.root = self.root.children[move_choice]

    def receive_evaluations(self, evaluations):
        """
        array ->
        Update nodes with newly received evaluations
        """
        self.cur_node.evaluation = evaluations[0]
        self.cur_node.probabilities = evaluations[1]
        cur_evaluation = evaluations[0] * -1
        # Propagate evaluation to parent nodes
        while self.cur_node.parent is not None:
            self.cur_node.parent.evaluation += cur_evaluation
            cur_evaluation *= -1
            self.cur_node = self.cur_node.parent

    def search(self, node):
        """
        Node ->
        Search a node
        """

        t0 = time.time()
        times = [0, 0, 0, 0]

        self.iterations_done += 1

        # Terminal node
        if node.game.outcome is not None:
            node.evaluation = node.game.outcome
            node.searches += 1
            cur_evaluation = node.game.outcome * -1
            # Propagate evaluation to parent nodes
            while node.parent is not None:
                node.parent.evaluation += cur_evaluation
                cur_evaluation *= -1
                node = node.parent
            # No evaluation needed
            times[-1] = time.time() - t0
            return (False, times)

        # First move
        if node.probabilities is None:
            self.cur_node = node
            node.legal_moves = node.game.get_legal_moves()
            times[-1] = time.time() - t0
            return (True, node.game.get_vector(), times)

        node.searches += 1

        # First time searching this node
        if node.moves is None:
            node.moves = []
            for i, is_legal in enumerate(node.legal_moves):
                if is_legal == 0:
                    node.probabilities[i] = 0
                else:
                    node.moves.append(i)
            node.probabilities = node.probabilities / sum(node.probabilities)
            node.children = [None] * len(node.moves)
            node.visits = np.full(len(node.moves), 0)
            node.noise = self.rng.dirichlet(np.full(len(node.moves), 0.3))

        # Choose next node to visit
        max_value = -1
        move_choice = -1
        t3 = time.time()
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
        times[2] += time.time() - t3

        # Record visit
        node.visits[move_choice] += 1

        # Exploring new node
        if node.children[move_choice] is None:
            t2 = time.time()
            new_game = deepcopy(node.game)
            res = new_game.do_move(node.moves[move_choice])
            times[1] += time.time() - t2
            node.children[move_choice] = Node(
                new_game,
                node.depth + 1,
                node,
            )
            # Terminal state
            if new_game.outcome is not None:
                node.children[move_choice].evaluation = new_game.outcome
                # No evaluation needed
                return (False, times)
            # Save legal moves if not terminal state
            node.children[move_choice].legal_moves = res[1]
            # Prepare to update the new node with received evaluations
            self.cur_node = node.children[move_choice]
            # Request evaluations
            times[-1] = time.time() - t0
            return (True, self.cur_node.game.get_vector(), times)

        # Otherwise, explore child node normally
        res = self.search(node.children[move_choice])
        for i in range(len(res[-1])):
            res[-1][i] += times[i]
        res[-1][-1] = time.time() - t0
        return res
