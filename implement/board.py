import numpy as np
from copy import deepcopy
from implement.move import Move


class Board:
    """
    A board of spaces
    There are three 2D arrays indicating whether a piece is present
    """

    def __init__(self):
        # Three 4x4 boards
        self.spaces = [False] * 48
        # Frozen spaces, defult to only give 3 spaces to prevent symmetry
        self.frozen = [True] * 16
        self.frozen[0 * 4 + 0] = False
        self.frozen[0 * 4 + 1] = False
        self.frozen[1 * 4 + 1] = False
        self.lines = []

    def __str__(self):
        """
        -> str
        String representation of board
        """
        output = ""
        for i in range(4):
            for j in range(4):
                # Base
                if self.spaces[i * 12 + j * 3 + 0]:
                    output += "B"
                else:
                    output += "_"
                # Column
                if self.spaces[i * 12 + j * 3 + 1]:
                    output += "C"
                else:
                    output += "_"
                # Capital
                if self.spaces[i * 12 + j * 3 + 2]:
                    output += "A"
                else:
                    output += "_"
                # Frozen space
                if self.frozen[i * 4 + j]:
                    output += "#"
                else:
                    output += " "
            output += "\n"
        return output

    def __deepcopy__(self, memo):
        """Custom deepcopy"""
        result = Board.__new__(Board)
        result.spaces = self.spaces[:]
        result.frozen = self.frozen[:]
        result.lines = self.lines[:]

        return result

    def is_empty(self, row, col):
        """
        int,int(,int) -> bool
        Takes a row and column number
        Returns whether the corresponding space is empty
        If ptype is specified, only considers that slot
        """
        return not any(self.spaces[row * 12 + col * 3 : row * 12 + col * 3 + 3])

    def bottom(self, row, col):
        """
        int,int -> int
        Takes a row and column number
        Returns the lowest piece on the space
        Returns -1 if space is empty
        """
        for i in [0, 1, 2]:
            if self.spaces[row * 12 + col * 3 + i]:
                return i
        return -1

    def top(self, row, col):
        """
        int,int -> int
        Takes a row and column number
        Returns the highest piece on the space
        Returns -1 if space is empty
        """
        for i in [2, 1, 0]:
            if self.spaces[row * 12 + col * 3 + i]:
                return i
        return -1

    def can_place(self, ptype, row, col):
        """
        int,int,int -> bool
        Returns whether a piece of ptype
        can be placed at (row, col)
        """
        # Check if space is frozen
        if self.is_empty(row, col):
            return True
        if self.frozen[row * 4 + col]:
            return False
        # Base
        if ptype == 0:
            return False
        # Column
        if ptype == 1:
            # Check for base
            return (
                self.spaces[row * 12 + col * 3 + 0]
                and not self.spaces[row * 12 + col * 3 + 1]
            )
        # Capital
        # Check for column and absence of capital
        return (
            self.spaces[row * 12 + col * 3 + 1]
            and not self.spaces[row * 12 + col * 3 + 2]
        )

    def can_move(self, row1, col1, row2, col2):
        """
        int,int,int,int -> bool
        Returns whether it is legal to move
        the stack at (row1,col1) to (row2,col2)
        """
        # Empty spaces
        if self.is_empty(row1, col1) or self.is_empty(row2, col2):
            return False
        # Frozen spaces
        if self.frozen[row1 * 4 + col1] or self.frozen[row2 * 4 + col2]:
            return False
        return self.bottom(row1, col1) - self.top(row2, col2) == 1

    def is_legal_move(self, move):
        """
        Move -> bool
        Takes a Move
        Returns whether the move is legal
        """
        # Place
        if move.mtype:
            return self.can_place(move.ptype, move.row1, move.col1)
        # Move
        return self.can_move(move.row1, move.col1, move.row2, move.col2)

    def do_move(self, move):
        """
        Move->
        Takes a Move
        Does the move, if it is legal
        """
        # Reset which space is frozen
        for row in range(4):
            self.frozen[row * 4 : row * 4 + 4] = [False] * 4
        # Place
        if move.mtype:
            self.spaces[move.row1 * 12 + move.col1 * 3 + move.ptype] = True
            self.frozen[move.row1 * 4 + move.col1] = True
        # Move
        else:
            self.spaces[move.row2 * 12 + move.col2 * 3 + 0] = (
                self.spaces[move.row1 * 12 + move.col1 * 3 + 0]
                or self.spaces[move.row2 * 12 + move.col2 * 3 + 0]
            )
            self.spaces[move.row2 * 12 + move.col2 * 3 + 1] = (
                self.spaces[move.row1 * 12 + move.col1 * 3 + 1]
                or self.spaces[move.row2 * 12 + move.col2 * 3 + 1]
            )
            self.spaces[move.row2 * 12 + move.col2 * 3 + 2] = (
                self.spaces[move.row1 * 12 + move.col1 * 3 + 2]
                or self.spaces[move.row2 * 12 + move.col2 * 3 + 2]
            )
            self.spaces[
                move.row1 * 12 + move.col1 * 3 : move.row1 * 12 + move.col1 * 3 + 3
            ] = [0, 0, 0]
            self.frozen[move.row2 * 4 + move.col2] = True
        self.get_lines()

    def get_tops(self):
        """
        Board -> array
        Returns an arrays with the tops of the spaces
        """
        tops = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                tops[i][j] = self.top(i, j)
        return tops

    def get_tops_and_bottoms(self):
        """
        Board -> array,array
        Returns two arrays with the tops and bottoms of the board
        """
        tops = np.zeros((4, 4))
        bottoms = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                tops[i][j] = self.top(i, j)
                bottoms[i][j] = self.bottom(i, j)
        return tops, bottoms

    def get_lines(self):
        """
        Board ->
        Returns an array of bools for which lines are made
        """
        # first, get array of tops, split by type
        tops = self.get_tops()
        self.lines = []
        if tops[0, 1] == tops[0, 2] > -1:
            line_type = tops[0, 1]
            if tops[0, 1] == tops[0, 0]:
                if tops[0, 1] == tops[0, 3]:
                    self.lines.append(("r0b", line_type))
                else:
                    self.lines.append(("r0l", line_type))
            elif tops[0, 1] == tops[0, 3]:
                self.lines.append(("r0r", line_type))
        if tops[1, 1] == tops[1, 2] > -1:
            line_type = tops[1, 1]
            if tops[1, 1] == tops[1, 0]:
                if tops[1, 1] == tops[1, 3]:
                    self.lines.append(("r1b", line_type))
                else:
                    self.lines.append(("r1l", line_type))
            elif tops[1, 1] == tops[1, 3]:
                self.lines.append(("r1r", line_type))
        if tops[2, 1] == tops[2, 2] > -1:
            line_type = tops[2, 1]
            if tops[2, 1] == tops[2, 0]:
                if tops[2, 1] == tops[2, 3]:
                    self.lines.append(("r2b", line_type))
                else:
                    self.lines.append(("r2l", line_type))
            elif tops[2, 1] == tops[2, 3]:
                self.lines.append(("r2r", line_type))
        if tops[3, 1] == tops[3, 2] > -1:
            line_type = tops[3, 1]
            if tops[3, 1] == tops[3, 0]:
                if tops[3, 1] == tops[3, 3]:
                    self.lines.append(("r3b", line_type))
                else:
                    self.lines.append(("r3l", line_type))
            elif tops[3, 1] == tops[3, 3]:
                self.lines.append(("r3r", line_type))
        if tops[1, 0] == tops[2, 0] > -1:
            line_type = tops[1, 0]
            if tops[1, 0] == tops[0, 0]:
                if tops[1, 0] == tops[3, 0]:
                    self.lines.append(("c0b", line_type))
                else:
                    self.lines.append(("c0u", line_type))
            elif tops[1, 0] == tops[3, 0]:
                self.lines.append(("c0d", line_type))
        if tops[1, 1] == tops[2, 1] > -1:
            line_type = tops[1, 1]
            if tops[1, 1] == tops[0, 1]:
                if tops[1, 1] == tops[3, 1]:
                    self.lines.append(("c1b", line_type))
                else:
                    self.lines.append(("c1u", line_type))
            elif tops[1, 1] == tops[3, 1]:
                self.lines.append(("c1d", line_type))
        if tops[1, 2] == tops[2, 2] > -1:
            line_type = tops[1, 2]
            if tops[1, 2] == tops[0, 2]:
                if tops[1, 2] == tops[3, 2]:
                    self.lines.append(("c2b", line_type))
                else:
                    self.lines.append(("c2u", line_type))
            elif tops[1, 2] == tops[3, 2]:
                self.lines.append(("c2d", line_type))
        if tops[1, 3] == tops[2, 3] > -1:
            line_type = tops[1, 3]
            if tops[1, 3] == tops[0, 3]:
                if tops[1, 3] == tops[3, 3]:
                    self.lines.append(("c3b", line_type))
                else:
                    self.lines.append(("c3u", line_type))
            elif tops[1, 3] == tops[3, 3]:
                self.lines.append(("c3d", line_type))
        # Upper left to lower right long diagonal
        if tops[1, 1] == tops[2, 2] > -1:
            line_type = tops[1, 1]
            if tops[1, 1] == tops[0, 0]:
                if tops[1, 1] == tops[3, 3]:
                    self.lines.append(("d0b", line_type))
                else:
                    self.lines.append(("d0u", line_type))
            elif tops[1, 1] == tops[3, 3]:
                self.lines.append(("d0d", line_type))
        # Upper right to lower left long diagonal
        if tops[1, 2] == tops[2, 1] > -1:
            line_type = tops[1, 2]
            if tops[1, 2] == tops[0, 3]:
                if tops[1, 2] == tops[3, 0]:
                    self.lines.append(("d1b", line_type))
                else:
                    self.lines.append(("d1u", line_type))
            elif tops[1, 2] == tops[3, 0]:
                self.lines.append(("d1d", line_type))
        # Top left short diagonal
        if tops[0, 2] == tops[1, 1] == tops[2, 0] > -1:
            self.lines.append(("s0", tops[0, 2]))
        # Top right short diagonal
        if -1 < tops[0, 1] == tops[1, 2] == tops[2, 3]:
            self.lines.append(("s1", tops[0, 1]))
        # Bottom right short diagonal
        if tops[1, 3] == tops[2, 2] == tops[3, 1] > -1:
            self.lines.append(("s2", tops[1, 3]))
        # Bottom left short diagonal
        if tops[1, 0] == tops[2, 1] == tops[3, 2] > -1:
            self.lines.append(("s3", tops[1, 0]))
