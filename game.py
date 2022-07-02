import numpy as np
import time
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
        moves = []
        # Need to break / extend lines
        if len(lines) > 0:
            for line in lines:
                cur = []
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
                        cur.extend(
                            [
                                Move(False, target, 1, 0, source, 1),
                                Move(False, target, 2, 0, source, 2),
                                Move(False, source, 1, 0, target, 1),
                                Move(False, source, 2, 0, target, 2),
                            ]
                        )
                        if line[0][2] == "l":
                            cur.extend(
                                [
                                    Move(False, target, 0, 0, source, 0),
                                    Move(False, source, 0, 0, target, 0),
                                ]
                            )
                            # Extend
                            if self.board.top(source, 3) == line[1]:
                                cur.append(Move(False, source, 3, 0, target, 3))
                        elif line[0][2] == "r":
                            cur.extend(
                                [
                                    Move(False, target, 3, 0, source, 3),
                                    Move(False, source, 3, 0, target, 3),
                                ]
                            )
                            # Extend
                            if self.board.top(source, 0) == line[1]:
                                cur.append(Move(False, source, 0, 0, target, 0))
                    # Place moves, not capitals
                    if line[1] < 2:
                        cur.extend(
                            [
                                Move(True, target, 1, line[1] + 1),
                                Move(True, target, 2, line[1] + 1),
                            ]
                        )
                        if line[0][2] == "l":
                            cur.append(Move(True, target, 0, line[1] + 1))
                            # Extend
                            cur.append(Move(True, target, 3, line[1]))
                        elif line[0][2] == "r":
                            cur.append(Move(True, target, 3, line[1] + 1))
                            # Extend
                            cur.append(Move(True, target, 0, line[1]))
                if line[0][0] == "c":
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
                        cur.extend(
                            [
                                Move(False, 1, target, 0, 1, source),
                                Move(False, 2, target, 0, 2, source),
                                Move(False, 1, source, 0, 1, target),
                                Move(False, 2, source, 0, 2, target),
                            ]
                        )
                        if line[0][2] == "u":
                            cur.extend(
                                [
                                    Move(False, 0, target, 0, 0, source),
                                    Move(False, 0, source, 0, 0, target),
                                ]
                            )
                            # Extend
                            if self.board.top(3, source) == line[1]:
                                cur.append(Move(False, 3, source, 0, 3, target))
                        elif line[0][2] == "d":
                            cur.extend(
                                [
                                    Move(False, 3, target, 0, 3, source),
                                    Move(False, 3, source, 0, 3, target),
                                ]
                            )
                            # Extend
                            if self.board.top(0, source) == line[1]:
                                cur.append(Move(False, 0, source, 0, 0, target))
                    # Place moves, not capitals
                    if line[1] < 2:
                        cur.extend(
                            [
                                Move(True, 1, target, line[1] + 1),
                                Move(True, 2, target, line[1] + 1),
                            ]
                        )
                        if line[0][2] == "u":
                            cur.append(Move(True, 0, target, line[1] + 1))
                            # Extend
                            cur.append(Move(True, 3, target, line[1]))
                        elif line[0][2] == "d":
                            cur.append(Move(True, 3, target, line[1] + 1))
                            # Extend
                            cur.append(Move(True, 0, target, line[1]))
                if line[0][0] == "d":
                    # Flip column if necessary
                    f = lambda x: x
                    if line[0][1] == "1":
                        f = lambda x: 3 - x
                    # Move moves
                    cur.extend(
                        [
                            Move(False, 1, f(1), 0, 0, f(1)),
                            Move(False, 1, f(1), 0, 1, f(2)),
                            Move(False, 1, f(1), 0, 2, f(1)),
                            Move(False, 1, f(1), 0, 1, f(0)),
                            Move(False, 0, f(1), 0, 1, f(1)),
                            Move(False, 1, f(2), 0, 1, f(1)),
                            Move(False, 2, f(1), 0, 1, f(1)),
                            Move(False, 1, f(0), 0, 1, f(1)),
                        ]
                    )
                    if line[0][2] == "u":
                        cur.extend(
                            [
                                Move(False, 0, f(0), 0, 0, f(1)),
                                Move(False, 0, f(0), 0, 1, f(0)),
                                Move(False, 0, f(1), 0, 0, f(0)),
                                Move(False, 1, f(0), 0, 0, f(0)),
                            ]
                        )
                        # Extend
                        if self.board.top(2, f(3)) == line[1]:
                            cur.append(Move(False, 2, f(3), 0, 3, f(3)))
                        if self.board.top(3, f(2)) == line[1]:
                            cur.append(Move(False, 3, f(2), 0, 3, f(3)))
                    elif line[0][2] == "d":
                        cur.extend(
                            [
                                Move(False, 3, f(3), 0, 2, f(3)),
                                Move(False, 3, f(3), 0, 3, f(2)),
                                Move(False, 2, f(3), 0, 3, f(3)),
                                Move(False, 3, f(2), 0, 3, f(3)),
                            ]
                        )
                        # Extend
                        if self.board.top(0, f(1)) == line[1]:
                            cur.append(Move(False, 0, f(1), 0, 0, f(0)))
                        if self.board.top(1, f(0)) == line[1]:
                            cur.append(Move(False, 1, f(0), 0, 0, f(0)))
                    # Place moves, not capitals
                    if line[1] < 2:
                        cur.extend(
                            [
                                Move(True, 1, f(1), line[1] + 1),
                                Move(True, 2, f(2), line[1] + 1),
                            ]
                        )
                        if line[0][2] == "u":
                            cur.append(Move(True, 0, f(0), line[1] + 1))
                            # Extend
                            cur.append(Move(True, 3, f(3), line[1]))
                        elif line[0][2] == "d":
                            cur.append(Move(True, 3, f(3), line[1] + 1))
                            # Extend
                            cur.append(Move(True, 0, f(0), line[1]))
                # Short diagonals
                else:
                    # Flip column if necessary
                    f = lambda x: x
                    if line[0][1] in ["1", "3"]:
                        f = lambda x: 3 - x
                    # Shift down if necessary
                    s = lambda x: x
                    if line[0][1] in ["2", "3"]:
                        s = lambda x: x + 1
                    # Move moves
                    cur.extend(
                        [
                            Move(False, s(0), f(3), 0, s(0), f(2)),
                            Move(False, s(1), f(2), 0, s(0), f(2)),
                            Move(False, s(0), f(1), 0, s(0), f(2)),
                            Move(False, s(0), f(1), 0, s(1), f(1)),
                            Move(False, s(1), f(2), 0, s(1), f(1)),
                            Move(False, s(2), f(1), 0, s(1), f(1)),
                            Move(False, s(1), f(0), 0, s(1), f(1)),
                            Move(False, s(1), f(0), 0, s(2), f(0)),
                            Move(False, s(2), f(1), 0, s(2), f(0)),
                            Move(False, s(0), f(2), 0, s(0), f(3)),
                            Move(False, s(0), f(2), 0, s(1), f(2)),
                            Move(False, s(0), f(2), 0, s(0), f(1)),
                            Move(False, s(1), f(1), 0, s(0), f(1)),
                            Move(False, s(1), f(1), 0, s(1), f(2)),
                            Move(False, s(1), f(1), 0, s(2), f(1)),
                            Move(False, s(1), f(1), 0, s(1), f(0)),
                            Move(False, s(2), f(0), 0, s(1), f(0)),
                            Move(False, s(2), f(0), 0, s(2), f(1)),
                        ]
                    )
                    if line[0][1] in ["2", "3"]:
                        cur.extend(
                            [
                                Move(False, 1, f(2), 0, 0, f(2)),
                                Move(False, 0, f(2), 0, 1, f(2)),
                            ]
                        )
                    else:
                        cur.extend(
                            [
                                Move(False, 3, f(0), 0, 2, f(0)),
                                Move(False, 2, f(0), 0, 3, f(0)),
                            ]
                        )
                    # Place moves, not capitals
                    if line[1] < 2:
                        cur.extend(
                            [
                                Move(True, s(0), f(2), line[1] + 1),
                                Move(True, s(1), f(1), line[1] + 1),
                                Move(True, s(2), f(0), line[1] + 1),
                            ]
                        )
                # First line
                if len(moves) == 0:
                    moves = cur
                else:
                    moves = list(set(moves) & set(cur))
        else:
            # Place
            for ptype in range(3):
                if self.pieces[self.to_play][ptype] == 0:
                    continue
                for row in range(4):
                    for col in range(4):
                        moves.append(Move(True, row, col, ptype))
            # Move
            for row1 in range(4):
                for row2 in range(4):
                    if abs(row1 - row2) > 1:
                        continue
                    for col1 in range(4):
                        for col2 in range(4):
                            moves.append(Move(False, row1, col1, 0, row2, col2))
        legal_moves = []
        for move in moves:
            if self.is_legal(move):
                legal_moves.append(move)
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
