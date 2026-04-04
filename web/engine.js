/**
 * Corintho rules engine (ported from corintho-ai Game::getLegalMoves / doMove).
 * Used for random CPU moves and legal-move lists without the server.
 */
import { LINE_BREAKERS } from "./line_breakers.js";

/** @see corintho-ai util.h */
export const K_GAME_STATE_SIZE = 70;
export const K_NUM_MOVES = 96;

const kBase = 0;
const kColumn = 1;
const kCapital = 2;
const kFrozen = 3;

const RL = 0;
const RR = 1;
const RB = 2;
const CU = 3;
const CD = 4;
const CB = 5;
const D0U = 0;
const D0D = 1;
const D0B = 2;
const D1U = 3;
const D1D = 4;
const D1B = 5;
const S0 = 6;
const S1 = 7;
const S2 = 8;
const S3 = 9;

function sp(coord1, coord2, flip = false) {
  if (flip) return { row: coord2, col: coord1 };
  return { row: coord1, col: coord2 };
}

export class CorinthoGame {
  /** @param {Int8Array} board 64 cells: [row*4+col]*4 + layer */
  constructor(board, pieces, toPlay) {
    this.board = board;
    this.pieces = pieces;
    this.toPlay = toPlay;
  }

  static fromGameState(gs) {
    const board = new Int8Array(64);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        const cell = gs.board[i][j];
        const o = (i * 4 + j) * 4;
        board[o + kBase] = cell.pieces.base ? 1 : 0;
        board[o + kColumn] = cell.pieces.column ? 1 : 0;
        board[o + kCapital] = cell.pieces.capital ? 1 : 0;
        board[o + kFrozen] = cell.frozen ? 1 : 0;
      }
    }
    const pieces = new Int8Array(6);
    const pt = ["base", "column", "capital"];
    for (let p = 0; p < 3; p++) {
      pieces[p] = gs.players[0].pieceCounts[pt[p]];
      pieces[3 + p] = gs.players[1].pieceCounts[pt[p]];
    }
    return new CorinthoGame(board, pieces, gs.turn);
  }

  clone() {
    return new CorinthoGame(
      Int8Array.from(this.board),
      Int8Array.from(this.pieces),
      this.toPlay
    );
  }

  boardAt(space, pieceType) {
    return this.board[space.row * 16 + space.col * 4 + pieceType] !== 0;
  }

  frozen(space) {
    return this.board[space.row * 16 + space.col * 4 + kFrozen] !== 0;
  }

  empty(space) {
    return !(
      this.boardAt(space, kBase) ||
      this.boardAt(space, kColumn) ||
      this.boardAt(space, kCapital)
    );
  }

  top(space) {
    for (let pieceType = kCapital; pieceType >= kBase; pieceType--) {
      if (this.boardAt(space, pieceType)) return pieceType;
    }
    return -1;
  }

  bottom(space) {
    for (let pieceType = kBase; pieceType <= kCapital; pieceType++) {
      if (this.boardAt(space, pieceType)) return pieceType;
    }
    return 3;
  }

  setBoard(space, pieceType, state) {
    this.board[space.row * 16 + space.col * 4 + pieceType] = state ? 1 : 0;
  }

  setFrozen(space, state) {
    this.board[space.row * 16 + space.col * 4 + kFrozen] = state ? 1 : 0;
  }

  canPlace(move) {
    const pieceType = move.pieceType;
    const to = move.spaceTo;
    if (this.pieces[this.toPlay * 3 + pieceType] === 0) return false;
    if (this.empty(to)) return true;
    if (this.frozen(to)) return false;
    if (pieceType === kBase) return false;
    if (pieceType === kColumn) {
      return !(this.boardAt(to, kColumn) || this.boardAt(to, kCapital));
    }
    return !(
      this.boardAt(to, kCapital) ||
      (this.boardAt(to, kBase) && !this.boardAt(to, kColumn))
    );
  }

  canMove(move) {
    const from = move.spaceFrom;
    const to = move.spaceTo;
    if (this.empty(from) || this.empty(to)) return false;
    if (this.frozen(from) || this.frozen(to)) return false;
    return this.bottom(from) - this.top(to) === 1;
  }

  isLegalMove(moveId) {
    const move = decodeMove(moveId);
    return move.place ? this.canPlace(move) : this.canMove(move);
  }

  applyLine(line, legal) {
    const mask = LINE_BREAKERS[line];
    for (let i = 0; i < 96; i++) {
      if (legal[i] && mask[95 - i] !== "1") legal[i] = false;
    }
  }

  applyRowColLines(legal, isCol) {
    for (let i = 0; i < 4; i++) {
      const top0 = this.top(sp(i, 0, isCol));
      const top1 = this.top(sp(i, 1, isCol));
      const top2 = this.top(sp(i, 2, isCol));
      const top3 = this.top(sp(i, 3, isCol));
      if (top1 === -1 || top2 === -1) continue;
      if (top0 === top1 && top1 === top2 && top2 === top3) {
        this.applyLine((isCol ? CB : RB) * 12 + i * 3 + top0, legal);
        return true;
      }
      for (const extendCoord of [3, 0]) {
        if (
          top1 === top2 &&
          ((extendCoord === 3 && top0 === top1) ||
            (extendCoord === 0 && top2 === top3))
        ) {
          if (isCol && extendCoord === 0) {
            this.applyLine(CD * 12 + i * 3 + top1, legal);
          } else if (isCol && extendCoord === 3) {
            this.applyLine(CU * 12 + i * 3 + top1, legal);
          } else if (extendCoord === 0) {
            this.applyLine(RR * 12 + i * 3 + top1, legal);
          } else {
            this.applyLine(RL * 12 + i * 3 + top1, legal);
          }
          if (top1 === 2) {
            if (!this.boardAt(sp(0, extendCoord, isCol), kCapital)) {
              legal[encodeMove(sp(0, extendCoord, isCol), sp(1, extendCoord, isCol))] = false;
            }
            if (!this.boardAt(sp(1, extendCoord, isCol), kCapital)) {
              legal[encodeMove(sp(1, extendCoord, isCol), sp(0, extendCoord, isCol))] = false;
              legal[encodeMove(sp(1, extendCoord, isCol), sp(2, extendCoord, isCol))] = false;
            }
            if (!this.boardAt(sp(2, extendCoord, isCol), kCapital)) {
              legal[encodeMove(sp(2, extendCoord, isCol), sp(1, extendCoord, isCol))] = false;
              legal[encodeMove(sp(2, extendCoord, isCol), sp(3, extendCoord, isCol))] = false;
            }
            if (!this.boardAt(sp(3, extendCoord, isCol), kCapital)) {
              legal[encodeMove(sp(3, extendCoord, isCol), sp(2, extendCoord, isCol))] = false;
            }
          }
          return true;
        }
      }
    }
    return false;
  }

  applyLongDiagLines(legal) {
    for (const flip of [false, true]) {
      const top0 = this.top(sp(0, flip ? 3 : 0));
      const top1 = this.top(sp(1, flip ? 2 : 1));
      const top2 = this.top(sp(2, flip ? 1 : 2));
      const top3 = this.top(sp(3, flip ? 0 : 3));
      if (top1 === -1 || top2 === -1) continue;
      if (top0 === top1 && top1 === top2 && top2 === top3) {
        this.applyLine(72 + (flip ? D1B : D0B) * 3 + top1, legal);
        return true;
      }
      if (top0 === top1 && top1 === top2) {
        this.applyLine(72 + (flip ? D1U : D0U) * 3 + top1, legal);
        return true;
      }
      if (top1 === top2 && top2 === top3) {
        this.applyLine(72 + (flip ? D1D : D0D) * 3 + top1, legal);
        return true;
      }
    }
    return false;
  }

  applyShortDiagLines(legal) {
    let top1 = this.top(sp(1, 1));
    if (top1 !== -1 && top1 === this.top(sp(0, 2)) && top1 === this.top(sp(2, 0))) {
      this.applyLine(72 + S0 * 3 + top1, legal);
      return true;
    }
    top1 = this.top(sp(1, 2));
    if (top1 !== -1 && top1 === this.top(sp(0, 1)) && top1 === this.top(sp(2, 3))) {
      this.applyLine(72 + S1 * 3 + top1, legal);
      return true;
    }
    top1 = this.top(sp(2, 2));
    if (top1 !== -1 && top1 === this.top(sp(1, 3)) && top1 === this.top(sp(3, 1))) {
      this.applyLine(72 + S2 * 3 + top1, legal);
      return true;
    }
    top1 = this.top(sp(2, 1));
    if (top1 !== -1 && top1 === this.top(sp(1, 0)) && top1 === this.top(sp(3, 2))) {
      this.applyLine(72 + S3 * 3 + top1, legal);
      return true;
    }
    return false;
  }

  applyLines(legal) {
    let any = false;
    any |= this.applyRowColLines(legal, false);
    any |= this.applyRowColLines(legal, true);
    any |= this.applyLongDiagLines(legal);
    any |= this.applyShortDiagLines(legal);
    return any;
  }

  getLegalMoves() {
    const legal = new Array(96).fill(true);
    const isLines = this.applyLines(legal);
    for (let i = 0; i < 96; i++) {
      if (legal[i] && !this.isLegalMove(i)) legal[i] = false;
    }
    return { legal, isLines };
  }

  legalMoveIds() {
    const { legal } = this.getLegalMoves();
    const out = [];
    for (let i = 0; i < 96; i++) if (legal[i]) out.push(i);
    return out;
  }

  doMove(moveId) {
    const move = decodeMove(moveId);
    for (let row = 0; row < 4; row++) {
      for (let col = 0; col < 4; col++) {
        this.setFrozen({ row, col }, false);
      }
    }
    if (move.place) {
      this.pieces[this.toPlay * 3 + move.pieceType]--;
      this.setBoard(move.spaceTo, move.pieceType, true);
      this.setFrozen(move.spaceTo, true);
    } else {
      for (const pieceType of [kBase, kColumn, kCapital]) {
        this.setBoard(
          move.spaceTo,
          pieceType,
          this.boardAt(move.spaceFrom, pieceType) || this.boardAt(move.spaceTo, pieceType)
        );
        this.setBoard(move.spaceFrom, pieceType, false);
      }
      this.setFrozen(move.spaceTo, true);
    }
    this.toPlay = 1 - this.toPlay;
  }

  /**
   * Game::writeGameState — 70 floats for the neural net (side-to-move canonized).
   * @param {Float32Array} out
   * @param {number} offset
   */
  writeGameState(out, offset = 0) {
    const b = offset;
    for (let i = 0; i < 64; i++) {
      out[b + i] = this.board[i] ? 1.0 : 0.0;
    }
    for (let i = 0; i < 6; i++) {
      const idx = (this.toPlay * 3 + i) % 6;
      out[b + 64 + i] = this.pieces[idx] * 0.25;
    }
  }
}

export function decodeMove(moveId) {
  if (moveId >= 48) {
    const pieceType = Math.floor((moveId - 48) / 16);
    const r = Math.floor(((moveId % 16) / 4));
    const c = moveId % 4;
    return {
      place: true,
      pieceType,
      spaceTo: { row: r, col: c },
    };
  }
  if (moveId < 12) {
    return {
      place: false,
      spaceFrom: { row: Math.floor(moveId / 3), col: moveId % 3 },
      spaceTo: { row: Math.floor(moveId / 3), col: (moveId % 3) + 1 },
    };
  }
  if (moveId < 24) {
    return {
      place: false,
      spaceFrom: { row: Math.floor((moveId - 12) / 4), col: moveId % 4 },
      spaceTo: { row: Math.floor((moveId - 12) / 4) + 1, col: moveId % 4 },
    };
  }
  if (moveId < 36) {
    return {
      place: false,
      spaceFrom: { row: Math.floor((moveId - 24) / 3), col: (moveId % 3) + 1 },
      spaceTo: { row: Math.floor((moveId - 24) / 3), col: moveId % 3 },
    };
  }
  return {
    place: false,
    spaceFrom: { row: Math.floor((moveId - 36) / 4) + 1, col: moveId % 4 },
    spaceTo: { row: Math.floor((moveId - 36) / 4), col: moveId % 4 },
  };
}

function encodeMove(spaceFrom, spaceTo) {
  if (spaceFrom.col < spaceTo.col) {
    return spaceFrom.row * 3 + spaceFrom.col;
  }
  if (spaceFrom.row < spaceTo.row) {
    return 12 + spaceFrom.row * 4 + spaceFrom.col;
  }
  if (spaceFrom.col > spaceTo.col) {
    return 24 + spaceFrom.row * 3 + (spaceFrom.col - 1);
  }
  return 36 + (spaceFrom.row - 1) * 4 + spaceFrom.col;
}

/** @returns {null | { 'pre-result': string }} */
export function preResultIfCpuTurnTerminal(gs) {
  if (gs.turn !== 1) return null;
  const g = CorinthoGame.fromGameState(gs);
  const { legal, isLines } = g.getLegalMoves();
  const count = legal.filter(Boolean).length;
  if (count > 0) return null;
  if (isLines) return { "pre-result": "win" };
  return { "pre-result": "draw" };
}

/**
 * Random CPU response matching the old server payload shape.
 * @param {import('./game.js').GameState} gameState
 */
export function chooseRandomCpuMove(gameState) {
  const pre = preResultIfCpuTurnTerminal(gameState);
  if (pre) return pre;

  const g = CorinthoGame.fromGameState(gameState);
  const ids = g.legalMoveIds();
  if (ids.length === 0) {
    const { isLines } = g.getLegalMoves();
    return isLines
      ? { "pre-result": "win" }
      : { "pre-result": "draw" };
  }
  const moveId = ids[Math.floor(Math.random() * ids.length)];
  const next = g.clone();
  next.doMove(moveId);

  const after = next.getLegalMoves();
  const noMoves = !after.legal.some(Boolean);
  if (noMoves) {
    return {
      move: moveId,
      is_done: true,
      has_won: after.isLines,
      legal_moves: [],
    };
  }

  const playerLegal = [];
  for (let i = 0; i < 96; i++) if (after.legal[i]) playerLegal.push(i);

  return {
    move: moveId,
    is_done: false,
    has_won: false,
    legal_moves: playerLegal,
  };
}
