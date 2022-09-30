cdef class Move:
    """
    Represents a move in Corintho
    Implement in Cython
    """

    cpdef bool mtype # Move or Place
    cpdef int row1
    cpdef int col1
    cpdef int row2
    cpdef int col2
    cpdef int ptype

    cpdef __init__(self, int move_id):
        """
        int -> Move
        Single ID constructor
        """
        # Move
        self.mtype = False
        # Right
        if move_id < 12:
            self.row1 = move_id // 3
            self.col1 = move_id % 3
            self.row2 = self.row1
            self.col2 = self.col1 + 1
        # Down
        elif move_id < 24:
            self.row1 = (move_id - 12) // 4
            self.col1 = move_id % 4
            self.row2 = self.row1 + 1
            self.col2 = self.col1
        # Left
        elif move_id < 36:
            self.row1 = (move_id - 24) // 3
            self.col1 = move_id % 3 + 1
            self.row2 = self.row1
            self.col2 = self.col1 - 1
        # Up
        elif move_id < 48:
            self.row1 = (move_id - 36) // 4 + 1
            self.col1 = move_id % 4
            self.row2 = self.row1 - 1
            self.col2 = self.col1
        # Place
        else:
            self.mtype = True
            self.ptype = (move_id - 48) // 16
            self.row1 = (move_id % 16) // 4
            self.col1 = move_id % 4

    @staticmethod
    cpdef place(int ptype, int row, int col):
        """
        bool,int,int -> Move
        Get a place move
        """
        out = Move(0)
        out.mtype = True
        out.ptype = ptype
        out.row1 = row
        out.col1 = col
        return out

    @staticmethod
    cpdef move(int row1, int col1, int row2, int col2):
        """
        int,int,int,int -> Move
        Get a move move
        """
        out = Move(0)
        out.row1 = row1
        out.col1 = col1
        out.row2 = row2
        out.col2 = col2
        return out

    @staticmethod
    cpdef encode_place(int ptype, int row, int col):
        return int(48 + ptype * 16 + row * 4 + col)

    @staticmethod
    cpdef encode_move(int row1, int col1, int row2, int col2):
        # Right
        if col1 < col2:
            return row1 * 3 + col1
        # Down
        if row1 < row2:
            return 12 + row1 * 4 + col1
        # Left
        if col1 > col2:
            return 24 + row1 * 3 + (col1 - 1)
        # Up
        return 36 + (row1 - 1) * 4 + col1

    def __str__(self):
        """
        -> str
        String representation of move
        """
        output = ""
        # Place
        if self.mtype:
            output += "P"
            if self.ptype == 0:
                output += "B"
            elif self.ptype == 1:
                output += "C"
            elif self.ptype == 2:
                output += "A"
            output += str(self.row1)
            output += str(self.col1)
        # Move
        else:
            output += "M"
            output += str(self.row1)
            output += str(self.col1)
            if self.row2 - self.row1 == 1:
                output += "D"
            elif self.row2 - self.row1 == -1:
                output += "U"
            elif self.col2 - self.col1 == 1:
                output += "R"
            elif self.col2 - self.col1 == -1:
                output += "L"
        return output
