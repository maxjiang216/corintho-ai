class GameState {
  constructor() {
      this.board = Array.from({
              length: 4
          }, () =>
          Array.from({
              length: 4
          }, () => ({
              pieces: {
                  capital: false,
                  column: false,
                  base: false
              },
              frozen: false,
          }))
      );
      this.players = [{
              id: 0,
              pieceCounts: {
                  base: 4,
                  column: 4,
                  capital: 4
              }
          },
          {
              id: 1,
              pieceCounts: {
                  base: 4,
                  column: 4,
                  capital: 4
              }
          },
      ];
      this.turn = 0;
      this.legalMoves = Array.from({
          length: 48
      }, (_, i) => i + 48);
  }

  placePiece(pieceType, row, col) {
      // We don't need to check if the player has the piece, this will be handled by the engine
      this.board[row][col].pieces[pieceType] = true;
      this.players[this.turn].pieceCounts[pieceType]--;
      // Unfreeze all spaces
      for (let row = 0; row < this.board.length; row++) {
          for (let col = 0; col < this.board[row].length; col++) {
              this.board[row][col].frozen = false;
          }
      }
      // Freeze the target space
      this.board[row][col].frozen = true;
      this.turn = 1 - this.turn;
  }

  movePiece(sourceRow, sourceCol, targetRow, targetCol) {
      const sourceCell = this.board[sourceRow][sourceCol];
      const targetCell = this.board[targetRow][targetCol];

      const targetPieces = targetCell.pieces;

      for (let pieceType in sourceCell.pieces) {
          if (sourceCell.pieces[pieceType]) {
              targetPieces[pieceType] = true;
              sourceCell.pieces[pieceType] = false;
          }
      }

      // Unfreeze all spaces
      for (let row = 0; row < this.board.length; row++) {
          for (let col = 0; col < this.board[row].length; col++) {
              this.board[row][col].frozen = false;
          }
      }

      // Freeze the target space
      targetCell.frozen = true;

      this.turn = 1 - this.turn;
  }

  isLegalPlace(pieceType, row, col) {
      const moveId = 48 + ["base", "column", "capital"].indexOf(pieceType) * 16 + row * 4 + col;
      return this.legalMoves.includes(moveId);
  }

  isLegalMove(sourceRow, sourceCol, targetRow, targetCol) {
      // Make sure it is orthogonally adjacent
      if (Math.abs(sourceRow - targetRow) + Math.abs(sourceCol - targetCol) !== 1) {
          return false;
      }
      let moveId;
      // Right
      if (sourceCol < targetCol) {
          moveId = sourceRow * 3 + sourceCol;
      }
      // Down
      else if (sourceRow < targetRow) {
          moveId = 12 + sourceRow * 4 + sourceCol;
      }
      // Left
      else if (sourceCol > targetCol) {
          moveId = 24 + sourceRow * 3 + sourceCol - 1;
      }
      // Up
      else {
          moveId = 36 + (sourceRow - 1) * 4 + sourceCol;
      }
      return this.legalMoves.includes(moveId);
  }
}

export {
  GameState
};
