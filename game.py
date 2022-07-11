import numpy as np
from copy import deepcopy
from board import Board
from move import Move


class Game:
    """
    Represents Coritho game
    A board with player pieces, current player, and outcome
    """

    def __init__(self):
        self.board = Board()
        # Which player is playing
        self.to_play = 0
        # Number of pieces for each player
        self.pieces = [[4, 4, 4], [4, 4, 4]]
        # Outcome when game is done. 1 is first player win. None means the game is ongoing
        self.outcome = None

    def __str__(self):
        return (
            str(self.board) + "\n" + str(self.pieces) + "\n" + str(self.to_play) + "\n"
        )

    def is_legal(self, move):
        """
        Move -> bool
        Checks if move is legal
        """
        # Place
        if move.mtype:
            if self.pieces[self.to_play][move.ptype] == 0 or not self.board.can_place(
                move.row1, move.col1, move.ptype
            ):
                return False
            return True
        # Move
        return self.board.can_move(move.row1, move.col1, move.row2, move.col2)

    def get_legal_moves(self):
        """
        -> array
        Returns a list of all legal moves
        """

        lines = self.board.lines
        # More than 2 lines, no legal moves
        if len(lines) > 2:
            return []
        moves = np.ones(96)
        # Need to break / extend lines
        if len(lines) > 0:
            for line in lines:
                cur = np.zeros(96)
                if line[0][0] == "r":
                    target = int(line[0][1])
                    sources = []
                    if target == 0:
                        sources = [1]
                    elif target == 1:
                        sources = [0, 2]
                    elif target == 2:
                        sources = [1, 3]
                    # 3
                    else:
                        sources = [2]
                    # Move moves
                    for source in sources:
                        f = lambda x, y, z: x + 4 * y + z
                        if target > source:
                            f = lambda x, y, z: (48 - x) + 4 * (x - 1) + z
                        cur[f(12, target, 1)] = 1
                        cur[f(12, target, 2)] = 1
                        cur[f(36, source, 1)] = 1
                        cur[f(36, source, 2)] = 1
                        if line[0][2] == "l":
                            cur[f(12, target, 0)] = 1
                            cur[f(36, source, 0)] = 1
                            # Extend
                            if self.board.top(source, 3) == line[1]:
                                cur[f(36, source, 3)] = 1
                        elif line[0][2] == "r":
                            cur[f(12, target, 3)] = 1
                            cur[f(36, source, 3)] = 1
                            # Extend
                            if self.board.top(source, 0) == line[1]:
                                cur[f(36, source, 0)] = 1
                    if line[0][2] == "l":
                        cur[target * 3 + 2] = 1
                    elif line[0][2] == "r":
                        cur[24 + target * 3] = 1
                    # Place moves, not on capitals
                    if line[1] < 2:
                        cur[48 + (line[1] + 1) * 16 + target * 4 + 1] = 1
                        cur[48 + (line[1] + 1) * 16 + target * 4 + 2] = 1
                        if line[0][2] == "l":
                            cur[48 + (line[1] + 1) * 16 + target * 4] = 1
                        elif line[0][2] == "r":
                            cur[48 + (line[1] + 1) * 16 + target * 4 + 3] = 1
                    # Place capital, extend
                    if line[0][2] == "l":
                        cur[48 + line[1] * 16 + target * 4 + 3] = 1
                    elif line[0][2] == "r":
                        cur[48 + line[1] * 16 + target * 4] = 1
                    target = int(line[0][1])
                    sources = []
                    if target == 0:
                        sources = [1]
                    elif target == 1:
                        sources = [0, 2]
                    elif target == 2:
                        sources = [1, 3]
                    # 3
                    else:
                        sources = [2]
                    # Move moves
                    for source in sources:
                        f = lambda x, y, z: x + 3 * y + z
                        if target > source:
                            f = lambda x, y, z: (24 - x) + 3 * y + (z - 1)
                        cur[f(0, 1, target)] = 1
                        cur[f(0, 2, target)] = 1
                        cur[f(24, 1, source)] = 1
                        cur[f(24, 2, source)] = 1
                        if line[0][2] == "u":
                            cur[f(0, 0, target)] = 1
                            cur[f(24, 0, source)] = 1
                            # Extend
                            if self.board.top(3, source) == line[1]:
                                cur[f(24, 3, source)] = 1
                        elif line[0][2] == "d":
                            cur[f(0, 3, target)] = 1
                            cur[f(24, 3, source)] = 1
                            # Extend
                            if self.board.top(0, source) == line[1]:
                                cur[f(24, 0, source)] = 1
                    if line[0][2] == "u":
                        cur[12 + 2 * 3 + target] = 1
                    elif line[0][2] == "r":
                        cur[36 + 0 * 3 + target] = 1
                    # Place moves, not on capitals
                    if line[1] < 2:
                        cur[48 + 16 * (line[1] + 1) + 1 * 3 + target] = 1
                        cur[48 + (line[1] + 1) * 16 + 2 * 3 + target] = 1
                        if line[0][2] == "u":
                            cur[48 + (line[1] + 1) * 16 + 0 * 3 + target] = 1
                        elif line[0][2] == "d":
                            cur[48 + (line[1] + 1) * 16 + 3 * 3 + target] = 1
                    # Place capital, extend
                    if line[0][2] == "u":
                        cur[48 + line[1] * 16 + 3 * 3 + target] = 1
                    elif line[0][2] == "d":
                        cur[48 + line[1] * 16 + 0 * 3 + target] = 1
                elif line[0][0] == "d":
                    # Flip column if necessary
                    f = lambda x, y, z: x + y * 3 + z
                    g = lambda x: x
                    s = lambda x: x
                    if line[0][1] == "1":
                        f = lambda x, y, z: (24 - x) + y * 3 + (3 - z)
                        g = lambda x: 3 - x
                        s = lambda x: 3 - x
                    # Move moves
                    cur[36 + 1 * 4 + s(1)] = 1
                    cur[f(0, 1, 1)] = 1
                    cur[12 + 1 * 4 + s(1)] = 1
                    cur[f(24, 1, 1)] = 1
                    cur[12 + 0 * 4 + s(1)] = 1
                    cur[f(24, 1, 2)] = 1
                    cur[36 + 2 * 4 + s(1)] = 1
                    cur[f(0, 1, 0)] = 1
                    if line[0][2] == "u":
                        cur[f(0, 0, 0)] = 1
                        cur[12 + 0 * 4 + s(0)] = 1
                        cur[f(24, 0, 1)] = 1
                        cur[36 + 1 * 4 + s(0)] = 1
                        # Extend
                        if self.board.top(2, g(3)) == line[1]:
                            cur[12 + 2 * 4 + s(3)] = 1
                        if self.board.top(3, g(2)) == line[1]:
                            cur[f(12, 3, 2)] = 1
                    elif line[0][2] == "d":
                        cur[36 + 3 * 4 + s(3)] = 1
                        cur[f(24, 3, 3)] = 1
                        cur[12 + 2 * 4 + s(3)] = 1
                        cur[f(0, 3, 2)] = 1
                        # Extend
                        if self.board.top(0, g(1)) == line[1]:
                            cur[f(24, 0, 1)] = 1
                        if self.board.top(1, f(0)) == line[1]:
                            cur[f(0, 1, 0)] = 1
                    # Place moves, not capitals
                    if line[1] < 2:
                        cur[48 + (line[1] + 1) * 16 + 1 * 4 + s(1)] = 1
                        cur[48 + (line[1] + 1) * 16 + 2 * 4 + s(2)] = 1
                        if line[0][2] == "u":
                            cur[48 + (line[1] + 1) * 16 + 1 * 0 + s(0)] = 1
                        elif line[0][2] == "d":
                            cur[48 + (line[1] + 1) * 16 + 3 * 4 + s(3)] = 1
                    # Place capital, extend
                    if line[0][2] == "u":
                        cur[48 + line[1] * 16 + 3 * 4 + s(3)] = 1
                    elif line[0][2] == "d":
                        cur[48 + line[1] * 16 + 3 * 0 + s(0)] = 1
                # Short diagonals
                else:
                    # Flip coordinates if necessary
                    f = lambda x, y, z: x + y * 3 + z
                    g = lambda x: x
                    s = lambda x, y, z: x + y * 4 + z
                    t = lambda x: x
                    if line[0][1] in ["1", "2"]:
                        f = lambda x, y, z: (24 - x) + y * 3 + (3 - z)
                        g = lambda x: 3 - x
                    if line[0][1] in ["2", "3"]:
                        s = lambda x, y, z: (48 - x) + (3 - y) * 4 + z
                        t = lambda x: 3 - x
                    # Move moves
                    cur[f(24, t(0), g(3))] = 1
                    cur[s(36, t(1), g(2))] = 1
                    cur[f(0, t(0), g(1))] = 1
                    cur[s(12, t(0), g(1))] = 1
                    cur[f(24, t(1), g(2))] = 1
                    cur[s(36, t(2), g(1))] = 1
                    cur[f(0, t(1), g(0))] = 1
                    cur[s(12, t(1), g(0))] = 1
                    cur[f(24, t(2), g(1))] = 1
                    cur[f(0, t(0), g(2))] = 1
                    cur[s(12, t(0), g(2))] = 1
                    cur[f(24, t(0), g(2))] = 1
                    cur[s(36, t(1), g(1))] = 1
                    cur[f(0, t(1), g(1))] = 1
                    cur[s(12, t(1), g(1))] = 1
                    cur[f(24, t(1), g(1))] = 1
                    cur[s(36, t(2), g(0))] = 1
                    cur[f(0, t(2), g(0))] = 1
                    if line[0][1] in ["2", "3"]:
                        cur[36 + 1 * 4 + g(2)] = 1
                        cur[12 + 0 * 4 + g(2)] = 1
                    else:
                        cur[36 + 3 * 4 + g(0)] = 1
                        cur[12 + 2 * 4 + g(0)] = 1
                    # Place moves, not capitals
                    if line[1] < 2:
                        cur[48 + (line[1] + 1) * 16 + t(0) * 4 + g(2)] = 1
                        cur[48 + (line[1] + 1) * 16 + t(1) * 4 + g(1)] = 1
                        cur[48 + (line[1] + 1) * 16 + t(2) * 4 + g(0)] = 1
                # First line
                if len(moves) == 0:
                    moves = cur
                else:
                    for i, move in enumerate(moves):
                        if not move:
                            moves[i] = 0
        legal_moves = np.zeros(96)
        for i, move in enumerate(moves):
            if move and self.is_legal(Move(i)):
                legal_moves[i] = 1
        return legal_moves

    def do_move(self, move):
        """
        Move -> bool,float
        Does move
        Current player loses if move is illegal
        Returns if game is done and the outcome
        """
        # Illegal move
        # if not self.is_legal(move):
        #     self.outcome = 1 - self.to_play
        #     return self.outcome
        # Place, remove piece from arsenal
        if move.mtype:
            self.pieces[self.to_play][move.ptype] -= 1
        self.board.do_move(move)
        self.to_play = 1 - self.to_play
        # Previous player win
        if len(self.get_legal_moves()) == 0:
            if len(self.board.lines) > 0:
                self.outcome = 2 * self.to_play - 1
            else:
                self.outcome = 0
            return self.outcome
        return None

    def get_vector(self):
        """
        -> array
        Returns vector representation of game in canonical form
        """

        canonical_pieces = [
            self.pieces[self.to_play],
            self.pieces[1 - self.to_play],
        ]
        return np.concatenate(
            [
                np.array(self.board.spaces).flatten(),
                np.array(self.board.frozen).flatten(),
                np.array(canonical_pieces).flatten() / 4,
            ]
        )
