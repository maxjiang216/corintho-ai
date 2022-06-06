import numpy as np


class Board:
    """
    A board of spaces
    There are three 2D arrays indicating whether a piece is present
    """

    def __init__(self):
        # Three 4x4 boards
        self.spaces = np.zeros((3, 4, 4))
        # 4x4 board filled with 1s
        self.frozen = np.zeros((4, 4)) + 1
        self.frozen[0][0] = 0
        self.frozen[0][1] = 0
        self.frozen[1][1] = 0

    def __str__(self):
        """
        -> str
        String representation of board
        """
        output = ""
        for i in range(4):
            for j in range(4):
                # Base
                if self.spaces[0][i][j]:
                    output += "B"
                else:
                    output += "_"
                # Column
                if self.spaces[1][i][j]:
                    output += "C"
                else:
                    output += "_"
                # Capital
                if self.spaces[2][i][j]:
                    output += "A"
                else:
                    output += "_"
                # Frozen space
                if self.frozen[i][j]:
                    output += "#"
                else:
                    output += "!"
        return output

    def is_empty(self, row, col, ptype=None):
        """
        int,int(,int) -> bool
        Takes a row and column number
        Returns whether the corresponding space is empty
        If ptype is specified, only considers that slot
        """
        if ptype is None:
            return np.max(self.spaces[:, row, col]) == 0
        return self.spaces[ptype, row, col] == 0

    def clear(self, row, col):
        """
        int,int ->
        Takes a row and column number
        Clears the corresponding space
        """
        self.spaces[:, row, col] = 0

    def bottom(self, row, col):
        """
        int,int -> int
        Takes a row and column number
        Returns the lowest piece on the space
        Returns -1 if space is empty
        """
        for i in range(3):
            if self.spaces[i][row][col]:
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
            if self.spaces[i][row][col]:
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
            return np.max(self.spaces[:, row, col]) == 0
        # Column
        if ptype == 1:
            # Check for absence of column and capital
            return self.spaces[1][row][col] == 0 and self.spaces[2][row][col] == 0
        # Capital
        # Check for column and absence of capital
        return self.spaces[1][row][col] == 1 and self.spaces[2][row][col] == 0

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
        if self.is_legal_move(move):
            # Place
            if move.mtype:
                self.spaces[move.ptype][move.row1][move.col1] = 1

                self.frozen.fill(0)
                self.frozen[move.row1][move.col1] = 1
            # Move
            else:
                self.spaces[:, move.row2, move.col2] = (
                    self.spaces[:, move.row2, move.col2]
                    + self.spaces[:, move.row1, move.col1]
                )
                self.clear(move.row1, move.col1)

                self.frozen.fill(0)
                self.frozen[move.row2][move.col2] = 1

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

    def count_lines(self, space=None):
        """
        (int,(int,int)) -> int
        Returns the number of 3 or 4 unit lines
        If a space is specified, only counts lines including the space
        """
        # first, get array of tops, split by type
        raw_tops = self.get_tops()
        count = 0
        if space is None:
            # loop for each piece type
            for i in range(3):
                tops = raw_tops == i
                # check rows and columns
                for j in range(4):
                    if (tops[j, 1] and tops[j, 2]) and (tops[j, 0] or tops[j, 3]):
                        count += 1
                    if (tops[1, j] and tops[2, j]) and (tops[0, j] or tops[3, j]):
                        count += 1
                # test long diagonals
                if (tops[1, 1] and tops[2, 2]) and (tops[0, 0] or tops[3, 3]):
                    count += 1
                if (tops[1, 2] and tops[2, 1]) and (tops[0, 3] or tops[3, 0]):
                    count += 1
                # short diagonals
                if tops[0, 1] and tops[1, 2] and tops[2, 3]:
                    count += 1
                if tops[1, 0] and tops[2, 1] and tops[3, 2]:
                    count += 1
                if tops[0, 2] and tops[1, 1] and tops[2, 0]:
                    count += 1
                if tops[1, 3] and tops[2, 2] and tops[3, 1]:
                    count += 1
            return count
        row, col = space[0], space[1]
        ptype = raw_tops[row][col]
        if ptype < 0:
            return 0
        tops = raw_tops == ptype
        # row and column
        if (tops[row, 1] and tops[row, 2]) and (
            (tops[row, 0] and col != 3) or (tops[row, 3] and col != 0)
        ):
            count += 1
        if (tops[1, col] and tops[2, col]) and (
            (tops[0, col] and row != 3) or (tops[3, col] and row != 0)
        ):
            count += 1
        # test diagonals
        if row == col:
            if (tops[1, 1] and tops[2, 2]) and (
                (tops[0, 0] and row != 3) or (tops[3, 3] and row != 0)
            ):
                count += 1
        if row + col == 3:
            if (tops[1, 2] and tops[2, 1]) and (
                (tops[0, 3] and row != 3) or (tops[3, 0] and row != 0)
            ):
                count += 1
        # short diagonals
        if row + 1 == col:
            if tops[0, 1] and tops[1, 2] and tops[2, 3]:
                count += 1
        if row == col + 1:
            if tops[1, 0] and tops[2, 1] and tops[3, 2]:
                count += 1
        if row + col == 2:
            if tops[0, 2] and tops[1, 1] and tops[2, 0]:
                count += 1
        if row + col == 4:
            if tops[1, 3] and tops[2, 2] and tops[3, 1]:
                count += 1
        return count
