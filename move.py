class Move:
    def __init__(self, mtype, row1=0, col1=0, ptype=0, row2=0, col2=0):
        """
        int(,int,int,int,int,int)
        """
        self.mtype = mtype
        # Place
        if mtype == 0:
            self.ptype = ptype
            self.row1 = row1
            self.col1 = col1
        # Move
        elif mtype == 1:
            self.row1 = row1
            self.col1 = col1
            self.row2 = row2
            self.col2 = col2

    def __str__(self):
        """
        -> str
        String representation of move
        """
        output = ""
        # Place
        if self.mtype == 0:
            if self.ptype == 0:
                output += "B"
            elif self.ptype == 1:
                output += "C"
            elif self.ptype == 2:
                output += "A"
            output += str(self.row1)
            output += str(self.col1)
        # Move
        elif self.mtype == 1:
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
        # Pass
        else:
            output = "Pass"
        return output
