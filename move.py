class Move:
    """
    Represents a move in Corintho
    """

    def __init__(self, mtype, row1, col1, ptype=0, row2=0, col2=0):
        """
        bool,int,int(,int,int,int)
        """
        self.mtype = mtype
        # Place
        if mtype:
            self.ptype = int(ptype)
            self.row1 = row1
            self.col1 = col1
            self.row2 = 0
            self.col2 = 0
        # Move
        else:
            self.row1 = row1
            self.col1 = col1
            self.row2 = row2
            self.col2 = col2
            self.ptype = 0

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
