/**
 * Web Worker: ONNX Runtime Web + TrainMC (no main-thread blocking).
 *
 * Model I/O (from tf2onnx conversion of tflite_model.tflite):
 *   input:  "serving_default_input_1:0"  [batch, 70]  float32
 *   output: "StatefulPartitionedCall:1"  [batch, 96]  float32  ← policy probs
 *   output: "StatefulPartitionedCall:0"  [batch, 1]   float32  ← scalar value
 */
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.mjs";
import { CorinthoGame, K_GAME_STATE_SIZE, K_NUM_MOVES, preResultIfCpuTurnTerminal } from "./engine.js";
import { TrainMC } from "./trainmc.js";

// Tell ORT where its WASM binaries live (same CDN, same version).
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
// Single-threaded is safer in a module worker; enable if you want SIMD.
ort.env.wasm.numThreads = 1;

const MODEL_URL = new URL("./model.onnx", import.meta.url).href;
const INPUT_NAME = "serving_default_input_1:0";
const OUTPUT_POLICY = "StatefulPartitionedCall:1"; // [batch, 96]
const OUTPUT_VALUE  = "StatefulPartitionedCall:0"; // [batch, 1]

/** Mulberry32 PRNG */
function createRng(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), a | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (t >>> 0) >>> 0;
  };
}

let sessionPromise = null;

function loadSession() {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
    });
  }
  return sessionPromise;
}

/**
 * Run ONNX inference.
 * @param {Float32Array} batchFloats  batchSize * 70 contiguous floats
 * @param {number} batchSize
 * @returns {{ probsData: Float32Array, evalData: Float32Array }}
 */
async function runInference(batchFloats, batchSize) {
  const session = await loadSession();
  const inputTensor = new ort.Tensor("float32", batchFloats.slice(0, batchSize * K_GAME_STATE_SIZE), [batchSize, K_GAME_STATE_SIZE]);
  const feeds = { [INPUT_NAME]: inputTensor };
  const results = await session.run(feeds);

  const policyTensor = results[OUTPUT_POLICY];
  const valueTensor  = results[OUTPUT_VALUE];

  const probsData = new Float32Array(policyTensor.data);
  const evalData  = new Float32Array(valueTensor.data);

  return { probsData, evalData };
}

function buildResponseAfterCpuMove(gameBeforeCpu, moveId) {
  const g = CorinthoGame.fromGameState(gameBeforeCpu);
  g.doMove(moveId);
  const after = g.getLegalMoves();
  const noMoves = !after.legal.some(Boolean);
  if (noMoves) {
    return { move: moveId, is_done: true, has_won: after.isLines, legal_moves: [] };
  }
  const playerLegal = [];
  for (let i = 0; i < K_NUM_MOVES; i++) if (after.legal[i]) playerLegal.push(i);
  return { move: moveId, is_done: false, has_won: false, legal_moves: playerLegal };
}

self.onmessage = async (ev) => {
  const gameState = ev.data;

  try {
    const pre = preResultIfCpuTurnTerminal(gameState);
    if (pre) {
      self.postMessage({ type: "result", payload: pre });
      return;
    }

    const rng = createRng(Date.now() ^ 0x9e3779b9);
    const randomUint32 = () => rng();

    const buf = new Float32Array(16 * K_GAME_STATE_SIZE);
    /** Deeper search = stronger play; `maxSearches` is sent to the UI for the progress bar. */
    const maxSearches = 4000;
    const trainmc = new TrainMC({
      randomUint32,
      toEvalBuffer: buf,
      maxSearches,
      searchesPerEval: 16,
      cPuct: 1.0,
      epsilon: 0.25,
      testing: true,
    });

    const rootGame = CorinthoGame.fromGameState(gameState);
    trainmc.createRoot(rootGame, 0);

    let evalArr = null;
    let probsArr = null;
    let done = false;
    let lastProgressPost = 0;

    while (!done) {
      done = trainmc.doIteration(evalArr, probsArr);
      evalArr = null;
      probsArr = null;

      const nReq = trainmc.numRequests();
      if (nReq > 0) {
        trainmc.writeRequests(buf);
        const { probsData, evalData } = await runInference(buf, nReq);
        evalArr = evalData;
        probsArr = probsData;
      }

      const searches = trainmc.searchesDone;

      const t = performance.now();
      if (done || t - lastProgressPost >= 110) {
        lastProgressPost = t;
        self.postMessage({ type: "progress", searches, maxSearches, done });
      }
    }

    const moveId = trainmc.chooseMove();

    const payload = buildResponseAfterCpuMove(gameState, moveId);
    self.postMessage({ type: "result", payload });

  } catch (e) {
    self.postMessage({
      type: "error",
      message: e && e.message ? e.message : String(e),
      stack: e && e.stack ? e.stack : undefined,
    });
  }
};
