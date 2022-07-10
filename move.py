class Move:
    """
    Represents a move in Corintho
    """

    def __init__(self, id):
        """
        int -> Move
        Single ID constructor
        """
        # Place
        self.mtype = False
        # Right
        if id < 12:
            self.row1 = id // 3
            self.col1 = id % 3
            self.row2 = self.row1
            self.col2 = self.col1 + 1
        # Down
        elif id < 24:
            self.row1 = (id - 12) // 4
            self.col1 = id % 4
            self.row2 = self.row1 + 1
            self.col2 = self.col1
        # Left
        elif id < 36:
            self.row1 = (id - 24) // 3
            self.col1 = id % 3 + 1
            self.row2 = self.row1
            self.col2 = self.col1 - 1
        # Up
        elif id < 48:
            self.row1 = (id - 36) // 4 + 1
            self.col1 = id % 4
            self.row2 = self.row1 - 1
            self.col2 = self.col1
        # Place
        else:
            self.mtype = True
            self.ptype = (id - 48) // 16
            self.row = (id % 16) // 4
            self.row = id % 4

    @staticmethod
    def place(ptype, row, col):
        """
        bool,int,int -> Move
        Get a place move
        """
        out = Move(0)
        out.ptype = True
        out.row = row
        out.col = col
        return out

    @staticmethod
    def move(row1, col1, row2, col2):
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

    def __eq__(self, obj):
        if self.mtype != obj.mtype:
            return False
        # Place
        if self.mtype:
            return (
                self.ptype == obj.ptype
                and self.row1 == obj.row1
                and self.col1 == obj.col1
            )
        return (
            self.row1 == obj.row1
            and self.col1 == obj.col1
            and self.row2 == obj.row2
            and self.col2 == obj.col2
        )

    def __key(self):
        return (self.mtype, self.row1, self.col1, self.ptype, self.row2, self.col2)

    def __hash__(self):
        return hash(self.__key())

    def __lt__(self, obj):
        if self.mtype and not obj.mtype:
            return True
        if not self.mtype and obj.mtype:
            return False
        if self.row1 < obj.row1:
            return True
        if self.row1 > obj.row1:
            return False
        if self.col1 < obj.col1:
            return True
        if self.col1 > obj.col1:
            return False
        if self.mtype:
            if self.ptype < obj.ptype:
                return True
            return False
        if self.row2 < obj.row2:
            return True
        if self.row2 > obj.row2:
            return False
        if self.col2 < obj.col2:
            return True
        return False

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
