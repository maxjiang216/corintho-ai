from copy import deepcopy
cdef class Board:
    
    cdef int bottoms[16], tops[16]
    cdef bint frozen[16]
    cdef list lines

    def __init__(self)
    def __str__(self)
    def __deepcopy__(self, memo)
    cdef bint is_empty(self, int row, int col)
    cdef bint can_place(self, int ptype, int row, int col)
    cdef bint can_move(self, int row1, int col1, int row2, int col2)
    cdef bint is_legal_move(self, Move move)
    cdef do_move(self, Move move):
        """
        Move->
        Takes a Move
        Does the move, if it is legal
        """
        # Reset which space is frozen
        for i in range(16):
            self.frozen[i] = False
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
