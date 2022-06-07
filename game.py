from board import Board


class Game:
    def __init__(self):
        self.board = Board()
        self.players = [Player(), Player()]

    def stringRepresentation(self):
        return (
            self.board.stringRepresentation()
            + self.players[0].stringRepresentation()
            + self.players[1].stringRepresentation()
        )

    def getActionSize(self):
        return 96

    def print(self):
        self.board.print()
        self.players[0].print()
        self.players[1].print()

    def turn(self, move, toplay):
        """
        move,int ->
        Does a move, if possible
        player=0 or 1
        """
        if self.players[toplay].canPlay(move) and self.board.legalMove(move):
            self.players[toplay].play(move)
            self.board.doMove(move)
        else:
            self.board.print()
            self.players[toplay].print()
            move.print()
            print("Invalid move!")

    def nextState(self, move, toplay):
        """
        move,int -> board,player
        Simulates a move, if possible
        Returns the new board and player
        """
        tempGame = copy.deepcopy(self)

        if tempGame.players[toplay].canPlay(move) and tempGame.board.legalMove(move):
            tempGame.players[toplay].play(move)
            tempGame.board.doMove(move)
        else:
            tempGame.board.print()
            tempGame.players[toplay].print()
            move.print()
            print("Invalid move!")

        return tempGame

    def getValidMoves(self, toplay):
        """
        int -> list(move)
        Returns all the legal moves (ignoring wincon) for the player to play
        """
        # placements
        output = []
        for i in range(3):
            if self.players[toplay].canPlayNum(i):
                for j in range(4):
                    for k in range(4):
                        if self.board.canPlaceNum(j, k, i):
                            output.append(Move("p", ptype=i, row1=j, col1=k))
        for i in range(4):
            for j in range(4):
                if i > 0:
                    if self.board.canMoveNum(i, j, i - 1, j):
                        output.append(Move("m", row1=i, col1=j, row2=i - 1, col2=j))
                if j < 3:
                    if self.board.canMoveNum(i, j, i, j + 1):
                        output.append(Move("m", row1=i, col1=j, row2=i, col2=j + 1))
                if i < 3:
                    if self.board.canMoveNum(i, j, i + 1, j):
                        output.append(Move("m", row1=i, col1=j, row2=i + 1, col2=j))
                if j > 0:
                    if self.board.canMoveNum(i, j, i, j - 1):
                        output.append(Move("m", row1=i, col1=j, row2=i, col2=j - 1))
        if self.board.countLines() == 0:
            return output
        cleaned = []
        for element in output:
            if element.mtype == "p":
                row = element.row1
                col = element.col1
            if element.mtype == "m":
                row = element.row2
                col = element.col2
            newGame = self.nextState(element, toplay)
            newBoard = newGame.board
            if newBoard.countLines() == newBoard.countLines(row, col):
                cleaned.append(element)
        return cleaned

    def getGameEnded(self, toplay):
        """
         -> bool
        Returns whether the game is won if the player moves next
        """
        nextMoves = self.getValidMoves(toplay)
        if len(nextMoves) == 0:
            if self.board.countLines() == 0:
                return 1e-8
            return -1
        return 0
