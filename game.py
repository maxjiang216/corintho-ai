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
                        f = lambda x, y, z: x + 4 * y + z
                        if target > source:
                            f = lambda x, y, z: (48 - x) + 4 * (x - 1) + z
                        cur.extend(
                            [
                                f(12, target, 1),
                                f(12, target, 2),
                                f(36, source, 1),
                                f(36, source, 2),
                            ]
                        )
                        if line[0][2] == "l":
                            cur.extend(
                                [
                                    f(12, target, 0),
                                    f(36, source, 0),
                                ]
                            )
                            # Extend
                            if self.board.top(source, 3) == line[1]:
                                cur.append(f(36, source, 3))
                        elif line[0][2] == "r":
                            cur.extend(
                                [
                                    f(12, target, 3),
                                    f(36, source, 3),
                                ]
                            )
                            # Extend
                            if self.board.top(source, 0) == line[1]:
                                cur.append(f(36, source, 0))
                    if line[0][2] == "l":
                        cur.append(target * 3 + 2)
                    elif line[0][2] == "r":
                        cur.append(24 + target * 3)
                    # Place moves, not on capitals
                    if line[1] < 2:
                        cur.extend(
                            [
                                48 + (line[1] + 1) * 16 + target * 4 + 1,
                                48 + (line[1] + 1) * 16 + target * 4 + 2,
                            ]
                        )
                        if line[0][2] == "l":
                            cur.append(48 + (line[1] + 1) * 16 + target * 4)
                        elif line[0][2] == "r":
                            cur.append(48 + (line[1] + 1) * 16 + target * 4 + 3)
                    # Place capital, extend
                    if line[0][2] == "l":
                        cur.append(48 + line[1] * 16 + target * 4 + 3)
                    elif line[0][2] == "r":
                        cur.append(48 + line[1] * 16 + target * 4)
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
                        cur.extend(
                            [
                                f(0, 1, target),
                                f(0, 2, target),
                                f(24, 1, source),
                                f(24, 2, source),
                            ]
                        )
                        if line[0][2] == "u":
                            cur.extend(
                                [
                                    f(0, 0, target),
                                    f(24, 0, source),
                                ]
                            )
                            # Extend
                            if self.board.top(3, source) == line[1]:
                                cur.append(f(24, 3, source))
                        elif line[0][2] == "d":
                            cur.extend(
                                [
                                    f(0, 3, target),
                                    f(24, 3, source),
                                ]
                            )
                            # Extend
                            if self.board.top(0, source) == line[1]:
                                cur.append(f(24, 0, source))
                    if line[0][2] == "u":
                        cur.append(12 + 2 * 3 + target)
                    elif line[0][2] == "r":
                        cur.append(36 + 0 * 3 + target)
                    # Place moves, not on capitals
                    if line[1] < 2:
                        cur.extend(
                            [
                                48 + 16 * (line[1] + 1) + 1 * 3 + target,
                                48 + (line[1] + 1) * 16 + 2 * 3 + target,
                            ]
                        )
                        if line[0][2] == "u":
                            cur.append(48 + (line[1] + 1) * 16 + 0 * 3 + target)
                        elif line[0][2] == "d":
                            cur.append(48 + (line[1] + 1) * 16 + 3 * 3 + target)
                    # Place capital, extend
                    if line[0][2] == "u":
                        cur.append(48 + line[1] * 16 + 3 * 3 + target)
                    elif line[0][2] == "d":
                        cur.append(48 + line[1] * 16 + 0 * 3 + target)
                elif line[0][0] == "d":
                    # Flip column if necessary
                    f = lambda x, y, z: x + y * 3 + z
                    s = lambda x: x
                    if line[0][1] == "1":
                        f = lambda x: (24 - x) + y * 3 + ((3 - x) - 1)
                        s = lambda x: 3 - x
                    # Move moves
                    cur.extend(
                        [
                            36 + 1 * 4 + s(1),
                            f(0, 1, 1),
                            12 + 1 * 4 + s(1),
                            f(24, 1, 1),
                            12 + 0 * 4 + s(1),
                            f(24, 1, 2),
                            36 + 2 * 4 + s(1),
                            f(0, 1, 0),
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
                        elif line[0][2] == "d":
                            cur.append(Move(True, 3, f(3), line[1] + 1))
                    # Place capital, extend
                    if line[0][2] == "u":
                        cur.append(Move(True, 3, f(3), line[1]))
                    elif line[0][2] == "d":
                        cur.append(Move(True, 0, f(0), line[1]))
                # Short diagonals
                else:
                    # Flip column if necessary
                    f = lambda x: x
                    if line[0][1] in ["1", "2"]:
                        f = lambda x: 3 - x
                    # Shift down if necessary
                    s = lambda x: x
                    if line[0][1] in ["2", "3"]:
                        s = lambda x: 3 - x
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
