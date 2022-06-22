import numpy as np
import time


class Board:
    """
    A board of spaces
    There are three 2D arrays indicating whether a piece is present
    """

    GetLinesTime = 0
    CanMoveTime = 0
    CanPlaceTime = 0

    def __init__(self):
        # Three 4x4 boards
        self.spaces = np.full((4, 4, 3), False)
        # Frozen spaces, defult to only give 3 spaces to prevent symmetry
        self.frozen = np.full((4, 4), True)
        self.frozen[0][0] = False
        self.frozen[0][1] = False
        self.frozen[1][1] = False
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
            output += "\n"
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
        t = time.time()
        # Check if space is frozen
        if self.frozen[row][col]:
            Board.CanPlaceTime += time.time() - t
            return False
        if self.is_empty(row, col):
            Board.CanPlaceTime += time.time() - t
            return True
        # Base
        if ptype == 0:
            Board.CanPlaceTime += time.time() - t
            return False
        # Column
        if ptype == 1:
            Board.CanPlaceTime += time.time() - t
            # Check for base
            return self.spaces[row][col][0] and not self.spaces[row][col][1]
        # Capital
        # Check for column and absence of capital
        Board.CanPlaceTime += time.time() - t
        return self.spaces[row][col][1] and not self.spaces[row][col][2]

    def can_move(self, row1, col1, row2, col2):
        """
        int,int,int,int -> bool
        Returns whether it is legal to move
        the stack at (row1,col1) to (row2,col2)
        """
        t = time.time()
        # Frozen spaces
        if self.frozen[row1][col1] or self.frozen[row2][col2]:
            Board.CanMoveTime += time.time() - t
            return False
        # Orthogonally adjacent spaces
        if abs(row1 - row2) + abs(col1 - col2) != 1:
            Board.CanMoveTime += time.time() - t
            return False
        # Empty spaces
        if self.is_empty(row1, col1) or self.is_empty(row2, col2):
            Board.CanMoveTime += time.time() - t
            return False
        Board.CanMoveTime += time.time() - t
        return self.bottom(row1, col1) - self.top(row2, col2) == 1

    def is_top(self, row, col, ptype):
        """
        int,int,int -> bool
        Checks if space (row,col) has top ptype
        """
        # Piece above
        if any(self.spaces[row][col][ptype + 1 :]):
            return False
        if self.spaces[row][col][ptype]:
            return True

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
        self.frozen = np.full((4, 4), False)
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
            self.spaces[move.row1][move.col1] = [0, 0, 0]
            self.frozen[move.row2][move.col2] = True
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
        t = time.time()
        # first, get array of tops, split by type
        tops = self.get_tops()
        self.lines = []
        if tops[0, 1] == tops[0, 2] > -1:
            if tops[0, 1] == tops[0, 0]:
                if tops[0, 1] == tops[0, 3]:
                    self.lines.append(("r0b", tops[0, 1]))
                else:
                    self.lines.append(("r0l", tops[0, 1]))
            elif tops[0, 1] == tops[0, 3]:
                self.lines.append(("r0r", tops[0, 1]))
        if tops[1, 1] == tops[1, 2] > -1:
            if tops[1, 1] == tops[1, 0]:
                if tops[1, 1] == tops[1, 3]:
                    self.lines.append("r1b")
                else:
                    self.lines.append("r1l")
            elif tops[1, 1] == tops[1, 3]:
                self.lines.append("r1r")
        if tops[2, 1] == tops[2, 2] > -1:
            if tops[2, 1] == tops[2, 0]:
                if tops[2, 1] == tops[2, 3]:
                    self.lines.append("r2b")
                else:
                    self.lines.append("r2l")
            elif tops[2, 1] == tops[2, 3]:
                self.lines.append("r2r")
        if tops[3, 1] == tops[3, 2] > -1:
            if tops[3, 1] == tops[3, 0]:
                if tops[3, 1] == tops[3, 3]:
                    self.lines.append("r3b")
                else:
                    self.lines.append("r3l")
            elif tops[3, 1] == tops[3, 3]:
                self.lines.append("r3r")
        if tops[1, 0] == tops[2, 0] > -1:
            if tops[1, 0] == tops[0, 0]:
                if tops[1, 0] == tops[3, 0]:
                    self.lines.append("c0b")
                else:
                    self.lines.append("c0u")
            elif tops[1, 0] == tops[3, 0]:
                self.lines.append("c0d")
        if tops[1, 1] == tops[2, 1] > -1:
            if tops[1, 1] == tops[0, 1]:
                if tops[1, 1] == tops[3, 1]:
                    self.lines.append("c1b")
                else:
                    self.lines.append("c1u")
            elif tops[1, 1] == tops[3, 1]:
                self.lines.append("c1d")
        if tops[1, 2] == tops[2, 2] > -1:
            if tops[1, 2] == tops[0, 2]:
                if tops[1, 2] == tops[3, 2]:
                    self.lines.append("c2b")
                else:
                    self.lines.append("c2u")
            elif tops[1, 2] == tops[3, 2]:
                self.lines.append("c2d")
        if tops[1, 3] == tops[2, 3] > -1:
            if tops[1, 3] == tops[0, 3]:
                if tops[1, 3] == tops[3, 3]:
                    self.lines.append("c3b")
                else:
                    self.lines.append("c3u")
            elif tops[1, 3] == tops[3, 3]:
                self.lines.append("c3d")
        # Upper left to lower right long diagonal
        if tops[1, 1] == tops[2, 2] > -1:
            if tops[1, 1] == tops[0, 0]:
                if tops[1, 1] == tops[3, 3]:
                    self.lines.append("d0b")
                else:
                    self.lines.append("d0u")
            elif tops[1, 1] == tops[3, 3]:
                self.lines.append("d0d")
        # Upper right to lower left long diagonal
        if tops[1, 2] == tops[2, 1] > -1:
            if tops[1, 2] == tops[0, 3]:
                if tops[1, 2] == tops[3, 0]:
                    self.lines.append("d1b")
                else:
                    self.lines.append("d1u")
            elif tops[1, 2] == tops[3, 0]:
                self.lines.append("d1d")
        # Top left short diagonal
        if tops[0, 2] == tops[1, 1] == tops[2, 0] > -1:
            self.lines.append("s0")
        # Top right short diagonal
        if -1 < tops[0, 1] == tops[1, 2] == tops[2, 3]:
            self.lines.append("s1")
        # Bottom right short diagonal
        if tops[1, 3] == tops[2, 2] == tops[3, 1] > -1:
            self.lines.append("s2")
        # Bottom left short diagonal
        if tops[1, 0] == tops[2, 1] == tops[3, 2] > -1:
            self.lines.append("s3")

        Board.GetLinesTime += time.time() - t
