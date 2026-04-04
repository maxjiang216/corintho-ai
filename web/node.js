/**
 * MCTS tree node — port of corintho-ai Node (node.cpp / node.h).
 */
import { CorinthoGame, K_GAME_STATE_SIZE, K_NUM_MOVES } from "./engine.js";

export const RESULT_NONE = 0;
export const RESULT_LOSS = 1;
export const RESULT_DRAW = 2;
export const RESULT_WIN = 3;
export const DEDUCED_LOSS = 4;
export const DEDUCED_DRAW = 5;
export const DEDUCED_WIN = 6;

/** @see Node::kMaxProbability */
export const K_MAX_PROBABILITY = 511.0;

export class MCTSNode {
  /**
   * @param {CorinthoGame} game
   * @param {MCTSNode | null} parent
   * @param {number} childId move id that created this node from parent (0 for root-from-game)
   * @param {number} depth
   * @param {boolean} applyMove if true, apply childId to parent's game (child constructor)
   */
  constructor(game, parent, childId, depth, applyMove) {
    this.game = game.clone();
    if (applyMove) {
      this.game.doMove(childId);
    }
    this.parent = parent;
    this.childId = childId;
    this.depth = depth;
    this.nextSibling = null;
    this.firstChild = null;
    /** @type {{ moveId: number, prob: number }[] | null} */
    this.edges = null;
    this.denominator = 0.0;
    this.evaluation = 0.0;
    this.visits = 1;
    this.result = RESULT_NONE;
    this.allVisited = true;
    this.numLegalMoves = 0;

    this._initializeEdges();
  }

  /** Root from arbitrary position (no move applied). */
  static createRoot(game, depth = 0) {
    return new MCTSNode(game, null, 0, depth, false);
  }

  /** Child after move from parent position (nextSibling links into list). */
  static createChild(parentGame, parent, nextSibling, moveId, depth) {
    const n = new MCTSNode(parentGame, parent, moveId, depth, true);
    n.nextSibling = nextSibling;
    return n;
  }

  _initializeEdges() {
    const { legal, isLines } = this.game.getLegalMoves();
    let count = 0;
    for (let i = 0; i < K_NUM_MOVES; i++) if (legal[i]) count++;
    this.numLegalMoves = count;

    if (count === 0) {
      this.result = isLines ? RESULT_LOSS : RESULT_DRAW;
      return;
    }

    this.edges = [];
    for (let i = 0; i < K_NUM_MOVES; i++) {
      if (legal[i]) {
        this.edges.push({ moveId: i, prob: 0 });
      }
    }
  }

  terminal() {
    return this.result === RESULT_LOSS || this.result === RESULT_DRAW;
  }

  known() {
    return this.result !== RESULT_NONE;
  }

  won() {
    return this.result === DEDUCED_WIN;
  }

  lost() {
    return this.result === RESULT_LOSS || this.result === DEDUCED_LOSS;
  }

  drawn() {
    return this.result === RESULT_DRAW || this.result === DEDUCED_DRAW;
  }

  moveIdAt(i) {
    return this.edges[i].moveId;
  }

  probability(i) {
    if (this.denominator <= 0) return 0;
    return this.edges[i].prob * this.denominator;
  }

  setProbability(i, p) {
    this.edges[i].prob = p;
  }

  setDenominator(d) {
    this.denominator = d;
  }

  setEvaluation(v) {
    this.evaluation = v;
  }

  setResult(r) {
    this.result = r;
  }

  setAllVisited(v = true) {
    this.allVisited = v;
  }

  incrementVisits() {
    this.visits++;
  }

  decrementVisits() {
    this.visits--;
  }

  increaseEvaluation(d) {
    this.evaluation += d;
  }

  decreaseEvaluation(d) {
    this.evaluation -= d;
  }

  nullParent() {
    this.parent = null;
  }

  nullNextSibling() {
    this.nextSibling = null;
  }

  countNodes() {
    let n = 1;
    let c = this.firstChild;
    while (c !== null) {
      n += c.countNodes();
      c = c.nextSibling;
    }
    return n;
  }

  writeGameState(out, offset = 0) {
    this.game.writeGameState(out, offset);
  }
}
