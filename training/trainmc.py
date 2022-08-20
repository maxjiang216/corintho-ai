import numpy as np
from copy import deepcopy
from implement.move import Move


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
        self.legal_moves = None
        self.probabilities = None
        self.children = None
        self.visits = None
        self.noise = None
        self.noisy_probabilities = None


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

        # Result to request for evaluations
        if evaluations is not None:
            # Choose evaluation from correct neural net
            if self.player_num is not None:
                self.receive_evaluations(evaluations[self.player_num])
            else:
                self.receive_evaluations(evaluations)

        # If we have not done enough searches, search again
        if self.iterations_done < self.iterations:
            self.iterations_done += 1
            res = self.search(self.root)
            # Search until an evaluation is needed
            while not res[0] and self.iterations_done < self.iterations:
                self.iterations_done += 1
                res = self.search(self.root)
            if res[0]:
                return ("eval", res[1])
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
        move_stats = {
            "eval": self.root.evaluation / self.root.searches,
            "searches": self.root.searches,
        }
        for i, move in enumerate(self.root.moves):
            if self.root.children[i] is not None:
                move_stats[str(Move(move))] = {
                    "prob": self.root.probabilities[i],
                }
                move_stats[str(Move(move))]["searches"] = self.root.children[i].searches
                move_stats[str(Move(move))]["eval"] = (
                    self.root.children[i].evaluation / self.root.children[i].searches
                )
        final_ratio = np.zeros(96)
        total_visits = sum(
            self.root.visits
        )  # Might be different from self.iterations due to reusing tree
        for i, move in enumerate(self.root.moves):
            if self.root.visits[i] > 0:
                final_ratio[move] = self.root.visits[i] / total_visits
        move = self.root.moves[move_choice]
        self.root = self.root.children[move_choice]
        return ("move", move, move_choice, final_ratio, move_stats)

    def receive_opp_move(self, move_choice):
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
            )
            self.root.legal_moves = res[1]
        # Unexplored state
        elif self.root.children[move_choice] is None:
            res = self.root.game.do_move(self.root.moves[move_choice])
            self.root = Node(
                self.root.game,
                self.root.depth + 1,
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
        self.cur_node.evaluation = evaluations[0] * (-1) ** self.root.game.to_play
        self.cur_node.probabilities = evaluations[1]
        cur_evaluation = self.cur_node.evaluation * -1
        # Propagate evaluation to parent nodes
        while self.cur_node.parent is not None:
            self.cur_node.parent.evaluation += cur_evaluation
            cur_evaluation *= -1
            self.cur_node = self.cur_node.parent

    def choose_next(self, node):
        # Choose next node to visit
        max_value = -1
        move_choice = -1
        for i in range(len(node.moves)):
            u = 0
            # If not visited, set action value to 0
            if node.children[i] is None:
                u = node.noisy_probabilities[i] * np.sqrt(node.searches - 1)
            else:
                u = -1 * node.children[i].evaluation / node.children[i].searches + (
                    node.noisy_probabilities[i]
                    * np.sqrt(node.searches - 1)
                    / (node.children[i].searches + 1)
                )
            if u > max_value:
                max_value = u
                move_choice = i

        return move_choice

    def search(self, node):
        """
        Node ->
        Search a node
        """

        # Terminal node
        if node.game.outcome is not None:
            node.evaluation = node.game.outcome * (-1) ** node.game.to_play
            node.searches += 1
            cur_evaluation = node.evaluation * -1
            # Propagate evaluation to parent nodes
            while node.parent is not None:
                node.parent.evaluation += cur_evaluation
                cur_evaluation *= -1
                node = node.parent
            # No evaluation needed
            return (False,)

        # First move
        if node.probabilities is None:
            self.cur_node = node
            node.legal_moves = node.game.get_legal_moves()
            return (True, node.game.get_vector())

        node.searches += 1

        # First time searching this node
        if node.moves is None:
            # This should occur if we receive a move from an opponent to an unsearched state
            if node.probabilities is None:
                self.cur_node = node
                # Request evaluations
                return (True, self.cur_node.game.get_vector())
            node.moves = []
            probabilities = []
            for i, is_legal in enumerate(node.legal_moves):
                if is_legal == 1:
                    node.moves.append(i)
                    probabilities.append(node.probabilities[i])
            node.probabilities = np.array(probabilities) / sum(probabilities)
            node.children = [None] * len(node.moves)
            node.visits = np.full(len(node.moves), 0)
            node.noise = self.rng.dirichlet(np.full(len(node.moves), 0.3))
            node.noisy_probabilities = np.zeros(len(node.moves))
            for i in range(len(node.moves)):
                node.noisy_probabilities[i] = self.c_puct * (
                    (1 - self.epsilon) * node.probabilities[i]
                    + self.epsilon * node.noise[i]
                )

        move_choice = self.choose_next(node)

        # Record visit
        node.visits[move_choice] += 1

        # Exploring new node
        if node.children[move_choice] is None:
            new_game = deepcopy(node.game)
            res = new_game.do_move(node.moves[move_choice])
            node.children[move_choice] = Node(
                new_game,
                node.depth + 1,
                node,
            )
            # Terminal state
            if new_game.outcome is not None:
                node.children[move_choice].evaluation = new_game.outcome
                # No evaluation needed
                return (False,)
            # Save legal moves if not terminal state
            node.children[move_choice].legal_moves = res[1]
            # Prepare to update the new node with received evaluations
            self.cur_node = node.children[move_choice]
            # Request evaluations
            return (True, self.cur_node.game.get_vector())

        # Otherwise, explore child node normally
        return self.search(node.children[move_choice])
