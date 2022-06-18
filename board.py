import numpy as np


class Board:
    """
    A board of spaces
    There are three 2D arrays indicating whether a piece is present
    """

    def __init__(self):
        # Three 4x4 boards
        self.spaces = []
        for row in range(4):
            temp_row = []
            for col in range(4):
                temp_row.append([False] * 3)
            self.spaces.append(temp_row)
        # Frozen spaces, defult to only give 3 spaces to prevent symmetry
        self.frozen = []
        for row in range(4):
            self.frozen.append([True] * 4)
        self.frozen[0][0] = False
        self.frozen[0][1] = False
        self.frozen[1][1] = False

    def __str__(self):
        """
        -> str
        String representation of board
        """
        output = ""
        for i in range(4):
            for j in range(4):
                # Base
                if self.spaces[i][j][0]:
                    output += "B"
                else:
                    output += "_"
                # Column
                if self.spaces[i][j][1]:
                    output += "C"
                else:
                    output += "_"
                # Capital
                if self.spaces[i][j][2]:
                    output += "A"
                else:
                    output += "_"
                # Frozen space
                if self.frozen[i][j]:
                    output += "#"
                else:
                    output += " "
        return output

    def is_empty(self, row, col):
        """
        int,int(,int) -> bool
        Takes a row and column number
        Returns whether the corresponding space is empty
        If ptype is specified, only considers that slot
        """
        return not any(ptype for ptype in self.spaces[row][col])

    def clear(self, row, col):
        """
        int,int ->
        Takes a row and column number
        Clears the corresponding space
        """
        self.spaces[row][col] = [False] * 3

    def bottom(self, row, col):
        """
        int,int -> int
        Takes a row and column number
        Returns the lowest piece on the space
        Returns -1 if space is empty
        """
        for i in range(3):
            if self.spaces[row][col][i]:
                return i
        return -1

    def top(self, row, col):
        """
        int,int -> int
        Takes a row and column number
        Returns the highest piece on the space
        Returns -1 if space is empty
        """
        for i in reversed(range(3)):
            if self.spaces[row][col][i]:
                return i
        return -1

    def can_place(self, row, col, ptype):
        """
        int,int,int -> bool
        Returns whether a piece of ptype
        can be placed at (row, col)
        """
        # Check if space is frozen
        if self.frozen[row][col]:
            return False
        # Base
        if ptype == 0:
            # Only if space is empty
            return self.is_empty(row, col)
        # Column
        if ptype == 1:
            # Check for absence of column and capital
            return not self.spaces[row][col][1] and not self.spaces[row][col][2]
        # Capital
        # Check for column and absence of capital
        return self.spaces[row][col][1] and not self.spaces[row][col][2]

    def can_move(self, row1, col1, row2, col2):
        """
        int,int,int,int -> bool
        Returns whether it is legal to move
        the stack at (row1,col1) to (row2,col2)
        """
        # Frozen spaces
        if self.frozen[row1][col1] or self.frozen[row2][col2]:
            return False
        # Empty spaces
        if self.is_empty(row1, col1) or self.is_empty(row2, col2):
            return False
        # Orthogonally adjacent spaces
        if abs(row1 - row2) + abs(col1 - col2) != 1:
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
            return self.can_place(move.row1, move.col1, move.ptype)
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
            self.frozen[row] = [False] * 4
        # Place
        if move.mtype:
            self.spaces[move.row1][move.col1][move.ptype] = True
            self.frozen[move.row1][move.col1] = True
        # Move
        else:
            self.spaces[move.row2][move.col2][0] = (
                self.spaces[move.row1][move.col1][0]
                or self.spaces[move.row2][move.col2][0]
            )
            self.spaces[move.row2][move.col2][1] = (
                self.spaces[move.row1][move.col1][1]
                or self.spaces[move.row2][move.col2][1]
            )
            self.spaces[move.row2][move.col2][2] = (
                self.spaces[move.row1][move.col1][2]
                or self.spaces[move.row2][move.col2][2]
            )
            self.frozen[move.row2][move.col2] = True

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
        Board -> array
        Returns an array of bools for which lines are made
        """
        # first, get array of tops, split by type
        tops = self.get_tops()
        lines = np.zeros(12)
        if tops[0, 1] == tops[0, 2]:
            if tops[0, 1] == tops[0, 0] == tops[0, 3]:
                lines[0] = 2
            elif tops[0, 1] == tops[0, 0] or tops[0, 1] == tops[0, 3]:
                lines[0] = 1
        if tops[1, 1] == tops[1, 2]:
            if tops[1, 1] == tops[1, 0] == tops[1, 3]:
                lines[1] = 2
            elif tops[1, 1] == tops[1, 0] or tops[1, 1] == tops[1, 3]:
                lines[1] = 1
        if tops[2, 1] == tops[2, 2]:
            if tops[2, 1] == tops[2, 0] == tops[2, 3]:
                lines[2] = 2
            elif tops[2, 1] == tops[2, 0] or tops[2, 1] == tops[2, 3]:
                lines[2] = 1
        if tops[3, 1] == tops[3, 2]:
            if tops[3, 1] == tops[3, 0] == tops[3, 3]:
                lines[3] = 2
            elif tops[3, 1] == tops[3, 0] or tops[3, 1] == tops[3, 3]:
                lines[3] = 1
        if tops[1, 0] == tops[2, 0]:
            if tops[1, 0] == tops[0, 0] == tops[3, 0]:
                lines[4] = 2
            elif tops[1, 0] == tops[0, 0] or tops[1, 0] == tops[3, 0]:
                lines[4] = 1
        if tops[1, 1] == tops[2, 1]:
            if tops[1, 1] == tops[0, 1] == tops[3, 1]:
                lines[5] = 2
            elif tops[1, 1] == tops[0, 1] or tops[1, 1] == tops[3, 1]:
                lines[5] = 1
        if tops[1, 2] == tops[2, 2]:
            if tops[1, 2] == tops[0, 2] == tops[3, 2]:
                lines[6] = 2
            elif tops[1, 2] == tops[0, 2] or tops[1, 2] == tops[3, 2]:
                lines[6] = 1
        if tops[1, 3] == tops[2, 3]:
            if tops[1, 3] == tops[0, 3] == tops[3, 3]:
                lines[7] = 2
            elif tops[1, 3] == tops[0, 3] or tops[1, 3] == tops[3, 3]:
                lines[7] = 1
        if tops[1, 1] == tops[2, 2]:
            if tops[1, 1] == tops[0, 0] == tops[3, 3]:
                lines[8] = 2
            elif tops[1, 1] == tops[0, 0] or tops[1, 1] == tops[3, 3]:
                lines[8] = 1
        if tops[1, 2] == tops[2, 1]:
            if tops[1, 2] == tops[0, 3] == tops[3, 0]:
                lines[9] = 2
            elif tops[1, 2] == tops[0, 3] or tops[1, 2] == tops[3, 0]:
                lines[9] = 1
        if tops[0, 1] == tops[1, 2] == tops[2, 3]:
            lines[10] = 1
        if tops[1, 0] == tops[2, 1] == tops[3, 2]:
            lines[11] = 1
        if tops[0, 2] == tops[1, 1] == tops[2, 0]:
            lines[12] = 1
        if tops[1, 3] == tops[2, 2] == tops[3, 1]:
            lines[13] = 1

        return lines
