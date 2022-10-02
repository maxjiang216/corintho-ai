from copy import deepcopy
import numpy as np
from move import Move


class Board:
    """
    A board of spaces
    There are three 2D arrays indicating whether a piece is present
    """

    def __init__(self):
        # Three 4x4 boards
        self.bottoms = [3] * 16
        self.tops = [-1] * 16
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
                if self.bottoms[i * 4 + j] == 0:
                    output += "B"
                else:
                    output += "_"
                # Column
                if 0 <= self.bottoms[i * 4 + j] <= 1 and 1 <= self.tops[i * 4 + j] <= 2:
                    output += "C"
                else:
                    output += "_"
                # Capital
                if self.tops[i * 4 + j] == 2:
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
        result.bottoms = self.bottoms[:]
        result.tops = self.tops[:]
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
        return self.tops[row * 4 + col] == -1

    def can_place(self, ptype, row, col):
        """
        int,int,int -> bool
        Returns whether a piece of ptype
        can be placed at (row, col)
        """
        if self.is_empty(row, col):
            return True
        # Check if space is frozen
        if self.frozen[row * 4 + col]:
            return False
        # Base
        if ptype == 0:
            return False
        # Column
        if ptype == 1:
            # Check for base
            return self.tops[row * 4 + col] == 0
        # Capital
        # Check for column and absence of capital
        return self.tops == 1

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
        return self.bottoms[row1 * 4 + col1] - self.tops[row2 * 4 + col2] == 1

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
            # Change if originally empty
            self.bottoms[move.row1 * 4 + move.col1] = min(
                move.ptype, self.bottoms[move.row1 * 4 + move.col1]
            )
            self.tops[move.row1 * 4 + move.col1] = move.ptype
            self.frozen[move.row1 * 4 + move.col1] = True
        # Move
        else:
            self.tops[move.row2 * 4 + move.col2] = self.tops[move.row1 * 4 + move.col1]
            self.bottoms[move.row1 * 4 + move.col1] = 3
            self.tops[move.row1 * 4 + move.col1] = -1
            self.frozen[move.row2 * 4 + move.col2] = True
        self.get_lines()

    def get_lines(self):
        """
        Board ->
        Returns an array of bools for which lines are made
        """
        # first, get array of tops, split by type
        tops = self.tops
        self.lines = []
        if tops[0 * 4 + 1] == tops[0 * 4 + 2] > -1:
            line_type = tops[0 * 4 + 1]
            if tops[0 * 4 + 1] == tops[0 * 4 + 0]:
                if tops[0 * 4 + 1] == tops[0 * 4 + 3]:
                    self.lines.append(("r0b", line_type))
                else:
                    self.lines.append(("r0l", line_type))
            elif tops[0 * 4 + 1] == tops[0 * 4 + 3]:
                self.lines.append(("r0r", line_type))
        if tops[1 * 4 + 1] == tops[1 * 4 + 2] > -1:
            line_type = tops[1 * 4 + 1]
            if tops[1 * 4 + 1] == tops[1 * 4 + 0]:
                if tops[1 * 4 + 1] == tops[1 * 4 + 3]:
                    self.lines.append(("r1b", line_type))
                else:
                    self.lines.append(("r1l", line_type))
            elif tops[1 * 4 + 1] == tops[1 * 4 + 3]:
                self.lines.append(("r1r", line_type))
        if tops[2 * 4 + 1] == tops[2 * 4 + 2] > -1:
            line_type = tops[2 * 4 + 1]
            if tops[2 * 4 + 1] == tops[2 * 4 + 0]:
                if tops[2 * 4 + 1] == tops[2 * 4 + 3]:
                    self.lines.append(("r2b", line_type))
                else:
                    self.lines.append(("r2l", line_type))
            elif tops[2 * 4 + 1] == tops[2 * 4 + 3]:
                self.lines.append(("r2r", line_type))
        if tops[3 * 4 + 1] == tops[3 * 4 + 2] > -1:
            line_type = tops[3 * 4 + 1]
            if tops[3 * 4 + 1] == tops[3 * 4 + 0]:
                if tops[3 * 4 + 1] == tops[3 * 4 + 3]:
                    self.lines.append(("r3b", line_type))
                else:
                    self.lines.append(("r3l", line_type))
            elif tops[3 * 4 + 1] == tops[3 * 4 + 3]:
                self.lines.append(("r3r", line_type))
        if tops[1 * 4 + 0] == tops[2 * 4 + 0] > -1:
            line_type = tops[1 * 4 + 0]
            if tops[1 * 4 + 0] == tops[0 * 4 + 0]:
                if tops[1 * 4 + 0] == tops[3 * 4 + 0]:
                    self.lines.append(("c0b", line_type))
                else:
                    self.lines.append(("c0u", line_type))
            elif tops[1 * 4 + 0] == tops[3 * 4 + 0]:
                self.lines.append(("c0d", line_type))
        if tops[1 * 4 + 1] == tops[2 * 4 + 1] > -1:
            line_type = tops[1 * 4 + 1]
            if tops[1 * 4 + 1] == tops[0 * 4 + 1]:
                if tops[1 * 4 + 1] == tops[3 * 4 + 1]:
                    self.lines.append(("c1b", line_type))
                else:
                    self.lines.append(("c1u", line_type))
            elif tops[1 * 4 + 1] == tops[3 * 4 + 1]:
                self.lines.append(("c1d", line_type))
        if tops[1 * 4 + 2] == tops[2 * 4 + 2] > -1:
            line_type = tops[1 * 4 + 2]
            if tops[1 * 4 + 2] == tops[0 * 4 + 2]:
                if tops[1 * 4 + 2] == tops[3 * 4 + 2]:
                    self.lines.append(("c2b", line_type))
                else:
                    self.lines.append(("c2u", line_type))
            elif tops[1 * 4 + 2] == tops[3 * 4 + 2]:
                self.lines.append(("c2d", line_type))
        if tops[1 * 4 + 3] == tops[2 * 4 + 3] > -1:
            line_type = tops[1 * 4 + 3]
            if tops[1 * 4 + 3] == tops[0 * 4 + 3]:
                if tops[1 * 4 + 3] == tops[3 * 4 + 3]:
                    self.lines.append(("c3b", line_type))
                else:
                    self.lines.append(("c3u", line_type))
            elif tops[1 * 4 + 3] == tops[3 * 4 + 3]:
                self.lines.append(("c3d", line_type))
        # Upper left to lower right long diagonal
        if tops[1 * 4 + 1] == tops[2 * 4 + 2] > -1:
            line_type = tops[1 * 4 + 1]
            if tops[1 * 4 + 1] == tops[0 * 4 + 0]:
                if tops[1 * 4 + 1] == tops[3 * 4 + 3]:
                    self.lines.append(("d0b", line_type))
                else:
                    self.lines.append(("d0u", line_type))
            elif tops[1 * 4 + 1] == tops[3 * 4 + 3]:
                self.lines.append(("d0d", line_type))
        # Upper right to lower left long diagonal
        if tops[1 * 4 + 2] == tops[2 * 4 + 1] > -1:
            line_type = tops[1 * 4 + 2]
            if tops[1 * 4 + 2] == tops[0 * 4 + 3]:
                if tops[1 * 4 + 2] == tops[3 * 4 + 0]:
                    self.lines.append(("d1b", line_type))
                else:
                    self.lines.append(("d1u", line_type))
            elif tops[1 * 4 + 2] == tops[3 * 4 + 0]:
                self.lines.append(("d1d", line_type))
        # Top left short diagonal
        if tops[0 * 4 + 2] == tops[1 * 4 + 1] == tops[2 * 4 + 0] > -1:
            self.lines.append(("s0", tops[0 * 4 + 2]))
        # Top right short diagonal
        if -1 < tops[0 * 4 + 1] == tops[1 * 4 + 2] == tops[2 * 4 + 3]:
            self.lines.append(("s1", tops[0 * 4 + 1]))
        # Bottom right short diagonal
        if tops[1 * 4 + 3] == tops[2 * 4 + 2] == tops[3 * 4 + 1] > -1:
            self.lines.append(("s2", tops[1 * 4 + 3]))
        # Bottom left short diagonal
        if tops[1 * 4 + 0] == tops[2 * 4 + 1] == tops[3 * 4 + 2] > -1:
            self.lines.append(("s3", tops[1 * 4 + 0]))
