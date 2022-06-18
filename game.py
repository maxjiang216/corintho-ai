from board import Board
from move import Move
from copy import deepcopy


class Game:
    def __init__(self):
        self.board = Board()
        # Which player is playing
        self.to_play = 0
        # Number of pieces for each player
        self.pieces = [[4, 4, 4], [4, 4, 4]]
        # If game is done
        self.done = False
        # Outcome when game is done. 1 is first player win
        self.outcome = 0

    def __string__(self):
        return str(self.board)

    def do_turn(self):
        """ """
        pass

    def is_legal(self, move):
        """
        Move -> bool
        Checks if move is legal
        """
        lines = self.board.get_lines()
        # Place
        if move.mtype:
            if self.pieces[self.to_play][move.ptype] == 0 or not self.board.can_place(
                move.row1, move.col1, move.ptype
            ):
                return False
        # Move
        else:
            if self.board.can_move(move.row1, move.col1, move.row2, move.col2):
                return False

        # Check that all lines are broken or extended
        temp_board = deepcopy(self.board)
        temp_board.do_move(move)
        new_lines = temp_board.get_lines()
        for count, line in enumerate(lines):
            if line == new_lines[count] == 1:
                return False
            elif line == 2 and new_lines[count] != 0:
                return False
        return True

    def get_legal_moves(self):
        """
        -> array
        Returns a list of all legal moves
        """
        moves = []
        # Place
        for ptype in range(3):
            if self.pieces[self.to_play][ptype] == 0:
                continue
            for row in range(4):
                for col in range(4):
                    moves.append(Move(True, row, col, ptype))
        # Move
        for row1 in range(4):
            for row2 in range(4):
                if abs(row1 - row2) > 1:
                    continue
                for col1 in range(4):
                    for col2 in range(4):
                        moves.append(Move(True, row, col, ptype))

        legal_moves = []
        for move in moves:
            if self.is_legal(move):
                legal_moves.append(move)

        return legal_moves

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
