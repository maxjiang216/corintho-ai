import numpy as np


class Board:
    """
    A board of spaces
    There are three 2D arrays indicating whether a piece is present
    """

    def __init__(self):
        self.spaces = np.zeros((3, 4, 4))
        self.frozen = np.zeros((4, 4)) + 1
        self.frozen[0][0] = 0
        self.frozen[0][1] = 0
        self.frozen[1][1] = 0

    def stringRepresentation(self):
        output = ""
        for i in range(4):
            for j in range(4):
                if self.spaces[0][i][j]:
                    output += "B"
                else:
                    output += "_"
                if self.spaces[1][i][j]:
                    output += "C"
                else:
                    output += "_"
                if self.spaces[2][i][j]:
                    output += "A"
                else:
                    output += "_"
                if self.frozen[i][j]:
                    output += "#"
                else:
                    output += "!"
        return output

    def print(self):
        print("---------------------")
        for i in range(4):
            print("|", end="")
            for j in range(4):
                if self.spaces[0][i][j]:
                    print("B", end="")
                else:
                    print(" ", end="")
                if self.spaces[1][i][j]:
                    print("C", end="")
                else:
                    print(" ", end="")
                if self.spaces[2][i][j]:
                    print("A", end="")
                else:
                    print(" ", end="")
                if self.frozen[i][j]:
                    print("#", end="")
                else:
                    print(" ", end="")
                print("|", end="")
            print("\n---------------------")

    def isEmpty(self, row, col, ptype=None):
        """
        int,int(,int) -> bool
        Takes a row and column number
        Returns whether the corresponding space is empty
        If ptype is specified, only considers that slot
        """
        if ptype == None:
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

    def canPlaceNum(self, row, col, ptype):
        """
        canplace, but with ints
        """
        if self.frozen[row][col]:
            return False
        if self.isEmpty(row, col):
            return True
        if ptype == 0:
            return False
        return self.isEmpty(row, col, ptype) and not self.isEmpty(row, col, ptype - 1)

    def canMoveNum(self, row1, col1, row2, col2):
        if self.frozen[row1][col1] or self.frozen[row2][col2]:
            return False
        if self.isEmpty(row1, col1) or self.isEmpty(row2, col2):
            return False
        if abs(row1 - row2) + abs(col1 - col2) != 1:
            return False

        return self.bottom(row1, col1) - self.top(row2, col2) == 1

    def legalMove(self, move):
        """
        Move -> bool
        Takes a Move
        Returns whether the move is legal
        """
        if move.mtype == "p":
            return self.canPlaceNum(move.row1, move.col1, move.ptype)
        if move.mtype == "m":
            return self.canMoveNum(move.row1, move.col1, move.row2, move.col2)

    def doMove(self, move):
        """
        Move->
        Takes a Move
        Does the move, if it is legal
        """
        if self.legalMove(move):
            if move.mtype == "p":
                self.spaces[move.ptype][move.row1][move.col1] = 1

                self.frozen.fill(0)
                self.frozen[move.row1][move.col1] = 1
            if move.mtype == "m":
                self.spaces[:, move.row2, move.col2] = (
                    self.spaces[:, move.row2, move.col2]
                    + self.spaces[:, move.row1, move.col1]
                )
                self.clear(move.row1, move.col1)

                self.frozen.fill(0)
                self.frozen[move.row2][move.col2] = 1

    def topsAndBottoms(self):
        """
        Board -> array,array
        Takes a board
        returns two arrays with the tops and bottoms of the board
        """
        tops = np.zeros((4, 4))
        bottoms = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                tops[i][j] = self.top(i, j)
                bottoms[i][j] = self.bottom(i, j)
        return tops, bottoms

    def countLines(self, row=None, col=None):
        """
        (int,int) -> int
        Returns the number of 3 or 4 unit lines
        If a space is specified, only counts lines including the space
        """
        # first, get array of tops, split by type
        rawTops, _ = self.topsAndBottoms()
        count = 0
        if row == None or col == None:
            # loop for each piece type
            for i in range(3):
                tops = rawTops == i
                for j in range(4):
                    if (tops[j, 1] and tops[j, 2]) and (tops[j, 0] or tops[j, 3]):
                        count += 1
                    if (tops[1, j] and tops[2, j]) and (tops[0, j] or tops[3, j]):
                        count += 1
                # test diagonals
                diag = tops.diagonal()
                if (diag[1] and diag[2]) and (diag[0] or diag[3]):
                    count += 1
                diag = np.fliplr(tops).diagonal()
                if (diag[1] and diag[2]) and (diag[0] or diag[3]):
                    count += 1
                # short diagonals
                diag = tops.diagonal(1)
                if diag[0] and diag[1] and diag[2]:
                    count += 1
                diag = tops.diagonal(-1)
                if diag[0] and diag[1] and diag[2]:
                    count += 1
                diag = np.fliplr(tops).diagonal(1)
                if diag[0] and diag[1] and diag[2]:
                    count += 1
                diag = np.fliplr(tops).diagonal(-1)
                if diag[0] and diag[1] and diag[2]:
                    count += 1
            return count
        ptype = rawTops[row][col]
        if ptype < 0:
            return 0
        tops = rawTops == ptype
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
            diag = tops.diagonal()
            if (diag[1] and diag[2]) and (
                (diag[0] and row != 3) or (diag[3] and row != 0)
            ):
                count += 1
        if row + col == 3:
            diag = np.fliplr(tops).diagonal()
            if (diag[1] and diag[2]) and (
                (diag[0] and row != 3) or (diag[3] and row != 0)
            ):
                count += 1
        # short diagonals
        if row + 1 == col:
            diag = tops.diagonal(1)
            if diag[0] and diag[1] and diag[2]:
                count += 1
        if row == col + 1:
            diag = tops.diagonal(-1)
            if diag[0] and diag[1] and diag[2]:
                count += 1
        if row + col == 2:
            diag = np.fliplr(tops).diagonal(1)
            if diag[0] and diag[1] and diag[2]:
                count += 1
        if row + col == 4:
            diag = np.fliplr(tops).diagonal(-1)
            if diag[0] and diag[1] and diag[2]:
                count += 1
        return count
