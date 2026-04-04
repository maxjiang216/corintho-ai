/**
 * Monte Carlo tree search — port of corintho-ai TrainMC (trainmc.cpp).
 * Web app uses testing=true (no opening temperature sampling).
 */
import { GAMMA_SAMPLES } from "./gamma_samples.js";
import {
  CorinthoGame,
  K_GAME_STATE_SIZE,
  K_NUM_MOVES,
} from "./engine.js";
import {
  MCTSNode,
  K_MAX_PROBABILITY,
  RESULT_NONE,
  RESULT_LOSS,
  RESULT_DRAW,
  DEDUCED_WIN,
  DEDUCED_LOSS,
  DEDUCED_DRAW,
} from "./node.js";

const K_NUM_OPENING_MOVES = 6;
const K_NEG_INF = -Infinity;

/** @typedef {{ type: "visited" | "new" | "none"; choice: number; insertAfter: MCTSNode | null; node: MCTSNode | null }} ChooseNextOutput */

export class TrainMC {
  /**
   * @param {object} opts
   * @param {() => number} opts.randomUint32 0..2^32-1
   * @param {Float32Array} opts.toEvalBuffer shared buffer searches_per_eval * 70
   */
  constructor(opts) {
    this.randomUint32 = opts.randomUint32;
    this.toEval = opts.toEvalBuffer;
    this.maxSearches = opts.maxSearches ?? 1600;
    this.searchesPerEval = opts.searchesPerEval ?? 16;
    this.cPuct = opts.cPuct ?? 1.0;
    this.epsilon = opts.epsilon ?? 0.25;
    /** testing=true: no opening stochasticity (matches DockerMC web constructor) */
    this.testing = opts.testing ?? true;

    /** @type {MCTSNode | null} */
    this.root = null;
    /** @type {MCTSNode | null} */
    this.cur = null;
    this.searchesDone = 0;
    /** @type {MCTSNode[]} */
    this.searched = [];
  }

  uninitialized() {
    return this.root === null;
  }

  numRequests() {
    return this.searched.length;
  }

  numNodes() {
    if (!this.root) return 0;
    return this.root.countNodes();
  }

  done() {
    return this.root ? this.root.terminal() : false;
  }

  drawn() {
    return this.root ? this.root.drawn() : false;
  }

  /**
   * @param {import('./engine.js').CorinthoGame} game
   * @param {number} depth
   */
  createRoot(game, depth = 0) {
    this.root = MCTSNode.createRoot(game, depth);
    this.cur = this.root;
  }

  writeRequests(gameStates) {
    for (let i = 0; i < this.searched.length; i++) {
      this.searched[i].writeGameState(gameStates, i * K_GAME_STATE_SIZE);
    }
  }

  /**
   * @param {Float32Array | null} eval_
   * @param {Float32Array | null} probs
   * @returns {boolean} true when this "turn" of search is complete
   */
  doIteration(eval_, probs) {
    if (this.toEval === null) throw new Error("toEval buffer missing");

    if (this.uninitialized()) {
      throw new Error("TrainMC: call createRoot before doIteration");
    }

    // Bootstrap: root needs first NN eval (matches trainmc.cpp lines 158–165)
    if (
      this.searchesDone === 0 &&
      this.root.visits === 1 &&
      this.root.allVisited
    ) {
      this.searchesDone = 1;
      this.cur = this.root;
      this.root.writeGameState(this.toEval, 0);
      this.searched.push(this.root);
      return false;
    }

    if (this.searched.length > 0) {
      if (!eval_ || !probs) {
        throw new Error("doIteration: expected eval and probs for pending batch");
      }
      this.receiveEval(eval_, probs);
    }

    while (
      this.searched.length < this.searchesPerEval &&
      this.searchesDone < this.maxSearches &&
      !this.root.known() &&
      !this.root.allVisited
    ) {
      this.search();
    }

    return (
      (this.searchesDone === this.maxSearches || this.root.known()) &&
      this.searched.length === 0
    );
  }

  getFilteredProbs(probsSlice, filteredOut) {
    const n = this.cur.numLegalMoves;
    let edgeIndex = 0;
    let sum = 0;
    for (let j = 0; j < K_NUM_MOVES; j++) {
      if (
        edgeIndex < n &&
        this.cur.moveIdAt(edgeIndex) === j
      ) {
        filteredOut[edgeIndex] = probsSlice[j];
        sum += filteredOut[edgeIndex];
        edgeIndex++;
        if (edgeIndex === n) break;
      }
    }
    const scalar = (1.0 / sum) * (1 - this.epsilon);
    for (let j = 0; j < n; j++) {
      filteredOut[j] *= scalar;
    }
  }

  generateDirichlet(dirichlet) {
    const n = this.cur.numLegalMoves;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const idx = this.randomUint32() % 1024;
      dirichlet[i] = GAMMA_SAMPLES[idx];
      sum += dirichlet[i];
    }
    const scalar = (1.0 / sum) * this.epsilon;
    for (let i = 0; i < n; i++) {
      dirichlet[i] *= scalar;
    }
  }

  setProbs(filteredProbs, dirichlet) {
    const n = this.cur.numLegalMoves;
    const weighted = new Array(n);
    let maxProb = 0;
    for (let j = 0; j < n; j++) {
      weighted[j] = filteredProbs[j] + dirichlet[j];
      if (weighted[j] > maxProb) maxProb = weighted[j];
    }
    const denom = K_MAX_PROBABILITY / maxProb;
    let finalSum = 0;
    for (let j = 0; j < n; j++) {
      const prob = Math.max(
        1,
        Math.round(weighted[j] * denom)
      );
      this.cur.setProbability(j, prob);
      finalSum += prob;
    }
    this.cur.setDenominator(1.0 / finalSum);
  }

  receiveEval(eval_, probs) {
    for (let i = 0; i < this.searched.length; i++) {
      this.cur = this.searched[i];
      const n = this.cur.numLegalMoves;
      const filteredProbs = new Float32Array(n);
      this.getFilteredProbs(
        probs.subarray(i * K_NUM_MOVES, (i + 1) * K_NUM_MOVES),
        filteredProbs
      );
      const dirichlet = new Float32Array(n);
      this.generateDirichlet(dirichlet);
      this.setProbs(filteredProbs, dirichlet);

      let curEval = eval_[i];
      while (this.cur.parent !== null) {
        this.cur.increaseEvaluation(curEval - 1.0);
        this.cur.setAllVisited(false);
        curEval *= -1.0;
        this.cur = this.cur.parent;
      }
      this.cur.increaseEvaluation(curEval - 1.0);
    }
    this.root.setAllVisited(false);
    this.searched = [];
  }

  chooseNext() {
    const vSqrt =
      this.cPuct * Math.sqrt(Math.max(1, this.cur.visits));
    let maxEval = K_NEG_INF;
    let choice = 0;
    let curChild = this.cur.firstChild;
    let edgeIndex = 0;
    /** @type {MCTSNode | null} */
    let prev = null;
    /** @type {MCTSNode | null} */
    let bestPrev = null;

    while (curChild !== null || edgeIndex < this.cur.numLegalMoves) {
      let u = K_NEG_INF;
      if (
        curChild !== null &&
        curChild.childId === this.cur.moveIdAt(edgeIndex)
      ) {
        if (
          (!curChild.known() || curChild.drawn()) &&
          !curChild.allVisited
        ) {
          if (curChild.drawn()) {
            u = this.cur.probability(edgeIndex) * vSqrt;
          } else {
            u =
              (-1.0 * curChild.evaluation) / curChild.visits +
              (this.cur.probability(edgeIndex) * vSqrt) /
                (curChild.visits + 1.0);
          }
        }
        prev = curChild;
        curChild = curChild.nextSibling;
      } else {
        u = this.cur.probability(edgeIndex) * vSqrt;
      }
      if (u > maxEval) {
        bestPrev = prev;
        maxEval = u;
        choice = this.cur.moveIdAt(edgeIndex);
      }
      edgeIndex++;
    }

    if (maxEval === K_NEG_INF) {
      return { type: "none", choice: -1, insertAfter: null, node: null };
    }
    if (bestPrev === null || bestPrev.childId !== choice) {
      return { type: "new", choice, insertAfter: bestPrev, node: null };
    }
    return { type: "visited", choice, insertAfter: null, node: bestPrev };
  }

  propagateTerminal() {
    let cur = this.cur;
    while (cur !== this.root) {
      if (cur.lost()) {
        cur = cur.parent;
        cur.setResult(DEDUCED_WIN);
      } else {
        cur = cur.parent;
        let curChild = cur.firstChild;
        let hasDraw = false;
        let edgeIndex = 0;
        while (curChild !== null) {
          if (
            curChild.childId !== cur.moveIdAt(edgeIndex) ||
            !curChild.known()
          ) {
            return;
          }
          if (curChild.drawn()) {
            hasDraw = true;
          }
          curChild = curChild.nextSibling;
          edgeIndex++;
        }
        if (edgeIndex < cur.numLegalMoves) return;
        if (hasDraw) {
          cur.setResult(DEDUCED_DRAW);
        } else {
          cur.setResult(DEDUCED_LOSS);
        }
      }
    }
  }

  search() {
    this.cur = this.root;
    this.searchesDone++;

    while (!this.cur.terminal()) {
      const res = this.chooseNext();
      this.cur.incrementVisits();
      this.cur.increaseEvaluation(1.0);

      if (res.type === "none") {
        this.cur.setAllVisited(true);
        while (this.cur.parent !== null) {
          this.cur.decrementVisits();
          this.cur.decreaseEvaluation(1.0);
          this.cur = this.cur.parent;
        }
        this.cur.decrementVisits();
        this.cur.decreaseEvaluation(1.0);
        this.searchesDone--;
        return;
      }

      if (res.type === "new" && res.insertAfter === null) {
        const child = MCTSNode.createChild(
          this.cur.game,
          this.cur,
          this.cur.firstChild,
          res.choice,
          this.cur.depth + 1
        );
        this.cur.firstChild = child;
        this.cur = child;
        break;
      }

      if (res.type === "new" && res.insertAfter !== null) {
        const child = MCTSNode.createChild(
          this.cur.game,
          this.cur,
          res.insertAfter.nextSibling,
          res.choice,
          this.cur.depth + 1
        );
        res.insertAfter.nextSibling = child;
        this.cur = child;
        break;
      }

      if (res.type === "visited") {
        this.cur = res.node;
      }
    }

    if (this.cur.terminal()) {
      this.propagateTerminal();
      let curEval = -1.0;
      if (this.cur.drawn()) curEval = 0.0;
      this.cur.setEvaluation(curEval);
      while (this.cur.parent !== null) {
        this.cur = this.cur.parent;
        this.cur.increaseEvaluation(curEval - 1.0);
        curEval *= -1.0;
      }
    } else {
      this.cur.setEvaluation(1.0);
      this.cur.writeGameState(
        this.toEval,
        this.searched.length * K_GAME_STATE_SIZE
      );
      this.searched.push(this.cur);
    }
    this.cur = this.root;
  }

  chooseHighProbMove() {
    let maxP = 0;
    let choice = 0;
    for (let i = 0; i < this.root.numLegalMoves; i++) {
      if (this.root.edges[i].prob > maxP) {
        maxP = this.root.edges[i].prob;
        choice = this.root.moveIdAt(i);
      }
    }
    return choice;
  }

  chooseMoveWon() {
    let cur = this.root.firstChild;
    let choice = 0;
    while (cur !== null) {
      if (cur.lost()) {
        choice = cur.childId;
        break;
      }
      cur = cur.nextSibling;
    }
    return choice;
  }

  chooseMoveLostDrawn() {
    let maxVisits = 0;
    let cur = this.root.firstChild;
    let choice = 0;
    while (cur !== null) {
      if (cur.visits > maxVisits && (this.root.lost() || !cur.won())) {
        choice = cur.childId;
        maxVisits = cur.visits;
      }
      cur = cur.nextSibling;
    }
    return choice;
  }

  chooseMoveNormal() {
    let maxVisits = 0;
    let maxEval = 0;
    let cur = this.root.firstChild;
    let choice = this.chooseHighProbMove();

    while (cur !== null) {
      if (!cur.won()) {
        let ev = cur.evaluation;
        if (cur.result === RESULT_DRAW || cur.result === DEDUCED_DRAW) {
          ev = 0.0;
        }
        if (
          cur.visits > maxVisits ||
          (cur.visits === maxVisits && ev > maxEval)
        ) {
          choice = cur.childId;
          maxVisits = cur.visits;
          maxEval = ev;
        }
      }
      cur = cur.nextSibling;
    }

    return choice;
  }

  /**
   * @returns {number} global move id
   */
  chooseMove() {
    if (this.searched.length !== 0) {
      throw new Error("chooseMove: pending evaluations");
    }
    if (this.root.won()) {
      return this.chooseMoveWon();
    }
    if (this.root.lost() || this.root.drawn()) {
      return this.chooseMoveLostDrawn();
    }
    if (this.root.depth < K_NUM_OPENING_MOVES && !this.testing) {
      return this.chooseMoveOpening();
    }
    return this.chooseMoveNormal();
  }

  chooseMoveOpening() {
    let visits = 0;
    let cur = this.root.firstChild;
    while (cur !== null) {
      if (!cur.won()) visits += cur.visits;
      cur = cur.nextSibling;
    }
    let choice = this.chooseHighProbMove();
    if (visits === 0) {
      return choice;
    }
    const target = this.randomUint32() % visits;
    let total = 0;
    cur = this.root.firstChild;
    while (cur !== null) {
      if (!cur.won()) {
        total += cur.visits;
        if (total > target) {
          choice = cur.childId;
          break;
        }
      }
      cur = cur.nextSibling;
    }
    return choice;
  }
}

export { K_GAME_STATE_SIZE, K_NUM_MOVES };
