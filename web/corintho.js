import { GameState } from "./game.js";
import { chooseRandomCpuMove } from "./engine.js";

/** @type {Worker} */
const cpuWorker = new Worker(new URL("./worker.js", import.meta.url), {
  type: "module",
});

/** Minimum time (ms) before the CPU move is applied — feels deliberate even when search is fast. */
const MIN_CPU_THINK_MS = 1200;

let gameState;

/** @type {{ page: number }} */
const rulesState = { page: 1 };

let cpuThinkingActive = false;
let cpuProgressReceived = false;

/** @type {{ kind: 'reserve', pieceType: string } | { kind: 'board', row: number, col: number } | null } */
let selection = null;

const PIECE_ORDER = ["base", "column", "capital"];

const THEME_STORAGE_KEY = "corintho-theme";

function toggleTheme() {
  const root = document.documentElement;
  const nextDark = !root.classList.contains("dark");
  root.classList.toggle("dark", nextDark);
  try {
    localStorage.setItem(THEME_STORAGE_KEY, nextDark ? "dark" : "light");
  } catch (_) {
    /* ignore */
  }
  syncThemeToggleUi();
}

function syncThemeToggleUi() {
  const btn = document.getElementById("theme-toggle");
  if (!btn) return;
  const dark = document.documentElement.classList.contains("dark");
  btn.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
  btn.setAttribute("title", dark ? "Light mode" : "Dark mode");
}

/** Rotating quote bank — attribution after em dash. */
const QUOTE_BANK = [
  "“Architecture is my delight, and putting up and pulling down, one of my favourite amusements.” — Thomas Jefferson, 3rd President of the United States",
  "“We shape our buildings; thereafter they shape us.” — Sir Winston Churchill, former Prime Minister of the United Kingdom",
  "“The wise woman builds her house, but with her own hands the foolish one tears hers down.” — Proverbs 14:1",
  "“By wisdom a house is built, and through understanding it is established” — Proverbs 24:3",
  "“Where were you when I laid the earth's foundation?” — God, to Job (Job 38:4)",
  "“Well building hath three conditions: firmness, commodity, and delight” — Vitruvius, Roman architect",
  "“Tactics flow from a positionally superior game” — Bobby Fischer, 11th World Chess Champion",
  "“To win against me, you must beat me three times: in the opening, the middlegame and the endgame” — Alexander Alekhine, 4th World Chess Champion",
  "“The whole entrepreneurial class is, as it were, in the position of a master-builder whose task it is to erect a building out of a limited supply of building materials” — Ludwig von Mises, Austrian economist",
  "“The threat is stronger than the execution.” — Aaron Nimzowitsch, Russian chess grandmaster",
  "“The stone that the builders rejected has become the cornerstone.” — Psalm 118:22",
  "“The general who wins a battle makes many calculations in his temple before the battle is fought.” — Sun Tzu, The Art of War",
];

const QUOTE_VISIBILITY_KEY = "corintho-quotes-visible";
const QUOTE_ROTATE_MS = 10000;

/** Full-bank shuffle once per page load; we cycle through this order deterministically. */
let quoteShuffleOrder = [];
/** Index into `quoteShuffleOrder` for the quote currently shown. */
let quoteOrderIndex = 0;
/** @type {ReturnType<typeof setInterval> | null} */
let quoteIntervalId = null;

/**
 * @param {string} s
 * @returns {{ full: string; body: string; attribution: string }}
 */
function parseQuoteEntry(s) {
  const parts = s.split(/\s*—\s*/);
  if (parts.length < 2) {
    return { full: s, body: s, attribution: "" };
  }
  return {
    full: s,
    body: parts[0].trim(),
    attribution: parts.slice(1).join(" — ").trim(),
  };
}

function shuffleQuoteBankOnce() {
  const arr = [...QUOTE_BANK];
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  quoteShuffleOrder = arr;
  quoteOrderIndex = 0;
}

function quoteEntryAtCurrentIndex() {
  if (quoteShuffleOrder.length === 0) {
    return { full: "", body: "", attribution: "" };
  }
  const s = quoteShuffleOrder[quoteOrderIndex % quoteShuffleOrder.length];
  return parseQuoteEntry(s);
}

function setQuoteDom(entry) {
  const bodyEl = document.getElementById("corintho-quote-body");
  const attrEl = document.getElementById("corintho-quote-attribution");
  if (bodyEl) bodyEl.textContent = entry.body;
  if (attrEl) {
    attrEl.textContent = entry.attribution;
    attrEl.style.display = entry.attribution ? "" : "none";
  }
}

function rotateQuote() {
  const el = document.getElementById("corintho-quote-text");
  if (!el) return;
  if (quoteShuffleOrder.length === 0) return;
  el.classList.add("opacity-0");
  window.setTimeout(() => {
    quoteOrderIndex = (quoteOrderIndex + 1) % quoteShuffleOrder.length;
    setQuoteDom(quoteEntryAtCurrentIndex());
    el.classList.remove("opacity-0");
  }, 500);
}

function stopQuoteRotation() {
  if (quoteIntervalId !== null) {
    clearInterval(quoteIntervalId);
    quoteIntervalId = null;
  }
}

function startQuoteRotation() {
  stopQuoteRotation();
  quoteIntervalId = window.setInterval(rotateQuote, QUOTE_ROTATE_MS);
}

function applyQuoteWidgetVisibility() {
  const panel = document.getElementById("corintho-quote-panel");
  const showBtn = document.getElementById("corintho-quote-show");
  if (!panel || !showBtn) return;
  let visible = true;
  try {
    visible = localStorage.getItem(QUOTE_VISIBILITY_KEY) !== "0";
  } catch (_) {
    /* ignore */
  }
  if (visible) {
    panel.classList.remove("hidden");
    panel.setAttribute("aria-hidden", "false");
    showBtn.classList.add("hidden");
    showBtn.classList.remove("flex");
    showBtn.setAttribute("aria-hidden", "true");
    startQuoteRotation();
  } else {
    panel.classList.add("hidden");
    panel.setAttribute("aria-hidden", "true");
    showBtn.classList.remove("hidden");
    showBtn.classList.add("flex");
    showBtn.setAttribute("aria-hidden", "false");
    stopQuoteRotation();
  }
}

function initQuoteWidget() {
  const wrap = document.getElementById("corintho-quote-text");
  const hideBtn = document.getElementById("corintho-quote-hide");
  const showBtn = document.getElementById("corintho-quote-show");
  if (!wrap || !hideBtn || !showBtn) return;

  shuffleQuoteBankOnce();
  setQuoteDom(quoteEntryAtCurrentIndex());
  applyQuoteWidgetVisibility();

  hideBtn.addEventListener("click", () => {
    try {
      localStorage.setItem(QUOTE_VISIBILITY_KEY, "0");
    } catch (_) {
      /* ignore */
    }
    applyQuoteWidgetVisibility();
  });

  showBtn.addEventListener("click", () => {
    try {
      localStorage.setItem(QUOTE_VISIBILITY_KEY, "1");
    } catch (_) {
      /* ignore */
    }
    setQuoteDom(quoteEntryAtCurrentIndex());
    applyQuoteWidgetVisibility();
  });
}

const PIECE_GLYPHS = {
  base: "layers",
  column: "view_column",
  capital: "crown",
};

/** Matches corintho-ai `move.cpp` Move::operator<< — rank = 4 - row (top row = 4). */
function getColName(col) {
  return "abcd"[col];
}

function rankNotation(row) {
  return String(4 - row);
}

function formatPlaceNotation(pieceType, row, col) {
  const letter =
    pieceType === "base" ? "B" : pieceType === "column" ? "C" : "A";
  return `${letter}${getColName(col)}${rankNotation(row)}`;
}

function formatMoveNotation(sourceRow, sourceCol, targetRow, targetCol) {
  let dir;
  if (targetCol < sourceCol) dir = "L";
  else if (targetCol > sourceCol) dir = "R";
  else if (targetRow < sourceRow) dir = "U";
  else dir = "D";
  return `${getColName(sourceCol)}${rankNotation(sourceRow)}${dir}`;
}

/** @type {{ side: 'you' | 'cpu', notation: string }[]} */
let moveHistory = [];

function pushHistory(side, notation) {
  moveHistory.push({ side, notation });
  renderHistoryList();
}

function renderHistoryList() {
  const ol = document.getElementById("move-history-list");
  const empty = document.getElementById("move-history-empty");
  if (!ol) return;
  ol.innerHTML = "";
  for (let i = 0; i < moveHistory.length; i++) {
    const { side, notation } = moveHistory[i];
    const li = document.createElement("li");
    li.textContent = `${side === "you" ? "You" : "CPU"}: ${notation}`;
    ol.appendChild(li);
  }
  if (empty) empty.style.display = moveHistory.length ? "none" : "";
}

function setActivePanel(panel) {
  const ids = {
    board: "panel-board",
    about: "panel-about",
    history: "panel-history",
  };
  for (const [key, id] of Object.entries(ids)) {
    const el = document.getElementById(id);
    if (!el) continue;
    const show = key === panel;
    el.classList.toggle("hidden", !show);
    if (show) el.removeAttribute("hidden");
    else el.setAttribute("hidden", "");
  }
  document.querySelectorAll(".sidebar-nav-btn[data-panel]").forEach((btn) => {
    const active = btn.dataset.panel === panel;
    btn.classList.toggle("sidebar-nav-btn--active", active);
    btn.setAttribute("aria-selected", active ? "true" : "false");
  });
}

function showNewGameOverlay() {
  const overlay = document.getElementById("overlay");
  const content = document.getElementById("overlay-content");
  if (!overlay || !content) return;
  content.innerHTML = `
    <h2 class="overlay-title" id="newgame-dialog-title">New game</h2>
    <p class="font-body text-sm text-stone-500 dark:text-stone-400 mb-6">Who takes the first turn?</p>
    <div class="overlay-btns flex-col sm:flex-row gap-3">
      <button type="button" id="newgame-first" class="btn btn-indigo">You play first</button>
      <button type="button" id="newgame-second" class="btn btn-gray">You play second</button>
    </div>
    <button type="button" id="newgame-cancel" class="mt-4 text-sm text-stone-500 dark:text-stone-400 hover:text-primary font-body bg-transparent border-0 cursor-pointer p-0">Cancel</button>
  `;
  overlay.classList.remove("hidden");
  overlay.setAttribute("aria-hidden", "false");
  content.setAttribute("aria-labelledby", "newgame-dialog-title");

  const go = async (humanFirst) => {
    hideRulesOverlay();
    await startNewGame(humanFirst);
  };
  document.getElementById("newgame-first")?.addEventListener("click", () => go(true));
  document.getElementById("newgame-second")?.addEventListener("click", () => go(false));
  document.getElementById("newgame-cancel")?.addEventListener("click", () => hideRulesOverlay());
}

async function startNewGame(humanFirst) {
  moveHistory = [];
  clearSelection();
  gameState = new GameState();
  if (!humanFirst) {
    gameState.turn = 1;
  }
  renderHistoryList();
  drawGame(gameState);
  setActivePanel("board");
  if (!humanFirst) {
    await runCpuResponse();
  }
}

async function runCpuResponse() {
  const moveData = await chooseCPUMove(gameState);
  if ("pre-result" in moveData) {
    endGame(moveData["pre-result"], gameState);
    return;
  }
  gameState.legalMoves = moveData["legal_moves"];
  moveData.move = normalizeMove(moveData.move);
  doCPUMove(gameState, moveData.move);
  if (moveData.is_done) {
    endGame(moveData.has_won ? "loss" : "draw", gameState);
  }
}

function isEmpty(cell) {
  return !cell.pieces.base && !cell.pieces.column && !cell.pieces.capital;
}

function canMoveFrom(row, col) {
  const cell = gameState.board[row][col];
  if (isEmpty(cell)) return false;
  const neighbors = [
    [row - 1, col],
    [row + 1, col],
    [row, col - 1],
    [row, col + 1],
  ];
  for (const [tr, tc] of neighbors) {
    if (tr < 0 || tr > 3 || tc < 0 || tc > 3) continue;
    if (gameState.isLegalMove(row, col, tr, tc)) return true;
  }
  return false;
}

function clearSelection() {
  selection = null;
}

function applySelectionHighlights() {
  document.querySelectorAll(".tile--selected, .tile--legal-target").forEach((el) => {
    el.classList.remove("tile--selected", "tile--legal-target");
  });
  document.querySelectorAll(".piece-container--selected").forEach((el) => {
    el.classList.remove("piece-container--selected");
  });

  if (!selection || gameState.turn !== 0) return;

  if (selection.kind === "reserve") {
    const btn = document.getElementById(selection.pieceType);
    btn?.closest(".piece-container")?.classList.add("piece-container--selected");
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        if (gameState.isLegalPlace(selection.pieceType, r, c)) {
          document
            .querySelector(`.tile[data-row="${r}"][data-col="${c}"]`)
            ?.classList.add("tile--legal-target");
        }
      }
    }
  } else {
    const { row, col } = selection;
    document
      .querySelector(`.tile[data-row="${row}"][data-col="${col}"]`)
      ?.classList.add("tile--selected");
    const neighbors = [
      [row - 1, col],
      [row + 1, col],
      [row, col - 1],
      [row, col + 1],
    ];
    for (const [tr, tc] of neighbors) {
      if (tr < 0 || tr > 3 || tc < 0 || tc > 3) continue;
      if (gameState.isLegalMove(row, col, tr, tc)) {
        document
          .querySelector(`.tile[data-row="${tr}"][data-col="${tc}"]`)
          ?.classList.add("tile--legal-target");
      }
    }
  }
}

function createPieceLayer(pieceType) {
  const layer = document.createElement("div");
  layer.className = `piece-layer piece-layer--${pieceType} piece-shadow`;
  const icon = document.createElement("span");
  icon.className = "material-symbols-outlined";
  icon.textContent = PIECE_GLYPHS[pieceType];
  icon.style.fontVariationSettings = "'FILL' 1";
  layer.appendChild(icon);
  return layer;
}

function drawGame(gs) {
  drawBoard(gs);
  drawPieceBank(gs);
  const board = document.getElementById("board");
  if (board && typeof board.animate === "function") {
    board.animate([{ opacity: 0.9 }, { opacity: 1 }], {
      duration: 260,
      easing: "cubic-bezier(0.22, 1, 0.36, 1)",
    });
  }
  applySelectionHighlights();
  if (gs.turn === 0 || gs.turn === 1) {
    drawTurn(gs);
  }
}

function drawBoard(gs) {
  const boardElement = document.getElementById("board");
  boardElement.innerHTML = "";
  boardElement.classList.toggle("corintho-board-grid--interactive", gs.turn === 0);

  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      const cell = gs.board[row][col];
      const tile = createTile(row, col, cell);
      boardElement.appendChild(tile);
    }
  }
}

function createTile(row, col, cell) {
  const letters = ["a", "b", "c", "d"];
  const tile = document.createElement("button");
  tile.type = "button";
  tile.classList.add("tile");
  tile.dataset.row = String(row);
  tile.dataset.col = String(col);
  tile.setAttribute(
    "aria-label",
    `Cell ${letters[col]}${rankNotation(row)}`
  );

  const hasStack = !isEmpty(cell);
  if (!hasStack) {
    tile.classList.add("tile--empty");
  } else {
    const stack = document.createElement("div");
    stack.className = "piece-stack piece-stack--board";
    for (const pieceType of PIECE_ORDER) {
      if (cell.pieces[pieceType]) {
        stack.appendChild(createPieceLayer(pieceType));
      }
    }
    tile.appendChild(stack);
  }

  if (cell.frozen) {
    tile.classList.add("frozen");
  }

  tile.addEventListener("click", () => handleTileClick(row, col));

  if ((row + col) % 2 === 0) {
    tile.classList.add("light");
  } else {
    tile.classList.add("dark");
  }

  return tile;
}

async function handleTileClick(row, col) {
  if (gameState.turn !== 0) return;

  const tile = gameState.board[row][col];

  if (!selection) {
    if (!isEmpty(tile) && canMoveFrom(row, col)) {
      selection = { kind: "board", row, col };
      applySelectionHighlights();
    }
    return;
  }

  if (selection.kind === "reserve") {
    if (gameState.isLegalPlace(selection.pieceType, row, col)) {
      const notation = formatPlaceNotation(selection.pieceType, row, col);
      gameState.placePiece(selection.pieceType, row, col);
      pushHistory("you", notation);
      clearSelection();
      await afterPlayerAction();
    }
    return;
  }

  if (selection.kind === "board") {
    if (selection.row === row && selection.col === col) {
      clearSelection();
      applySelectionHighlights();
      return;
    }
    if (gameState.isLegalMove(selection.row, selection.col, row, col)) {
      const notation = formatMoveNotation(
        selection.row,
        selection.col,
        row,
        col
      );
      gameState.movePiece(selection.row, selection.col, row, col);
      pushHistory("you", notation);
      clearSelection();
      await afterPlayerAction();
      return;
    }
    if (!isEmpty(tile) && canMoveFrom(row, col)) {
      selection = { kind: "board", row, col };
      applySelectionHighlights();
    }
  }
}

async function afterPlayerAction() {
  drawGame(gameState);
  await new Promise((r) => requestAnimationFrame(() => r()));
  await runCpuResponse();
}

function handleReserveClick(pieceType) {
  if (gameState.turn !== 0) return;
  if (gameState.players[0].pieceCounts[pieceType] <= 0) return;
  if (selection?.kind === "reserve" && selection.pieceType === pieceType) {
    clearSelection();
  } else {
    selection = { kind: "reserve", pieceType };
  }
  applySelectionHighlights();
}

function updateCounter(gs, pieceType, cpu) {
  let counterElement;
  if (cpu) {
    counterElement = document.getElementById(`counter-${pieceType}-cpu`);
  } else {
    counterElement = document.getElementById(`counter-${pieceType}`);
  }
  counterElement.textContent = gs.players[cpu ? 1 : 0].pieceCounts[pieceType];
}

function drawPieceBank(gs) {
  for (const pieceType of PIECE_ORDER) {
    updateCounter(gs, pieceType, true);
    updateCounter(gs, pieceType, false);
  }
  for (const pieceType of PIECE_ORDER) {
    const btn = document.getElementById(pieceType);
    if (btn) {
      const count = gs.players[0].pieceCounts[pieceType];
      btn.disabled = count <= 0;
      btn.classList.toggle("piece--disabled", count <= 0);
    }
  }
}

function drawTurn(gs) {
  const turnElement = document.getElementById("turn-counter");
  const turnArea = document.getElementById("turn-area");
  if (!turnElement) return;
  if (gs.turn === -1) {
    stopCpuThinking();
    return;
  }
  if (cpuThinkingActive) {
    return;
  }
  if (turnArea) turnArea.classList.remove("turn-area--thinking");
  turnElement.classList.remove("hidden");
  turnElement.textContent = `It's ${gs.turn === 0 ? "your" : "the CPU's"} turn!`;
}

function startCpuThinking() {
  const counter = document.getElementById("turn-counter");
  const panel = document.getElementById("cpu-thinking");
  const fill = document.getElementById("cpu-thinking-fill");
  const track = document.getElementById("cpu-thinking-track");
  const area = document.getElementById("turn-area");
  if (!counter || !panel || !fill) return;
  cpuThinkingActive = true;
  cpuProgressReceived = false;
  counter.classList.add("hidden");
  panel.classList.remove("hidden");
  panel.setAttribute("aria-hidden", "false");
  if (area) area.classList.add("turn-area--thinking");
  fill.classList.remove("cpu-thinking__fill--indeterminate");
  fill.style.width = "0%";
  if (track) {
    track.setAttribute("aria-valuenow", "0");
    track.setAttribute("aria-busy", "true");
  }
}

/**
 * @param {number} searches
 * @param {number} maxSearches
 * @param {boolean} [searchDone]
 */
function updateCpuThinking(searches, maxSearches, searchDone) {
  const fill = document.getElementById("cpu-thinking-fill");
  const track = document.getElementById("cpu-thinking-track");
  if (!fill || !maxSearches) return;
  cpuProgressReceived = true;
  let pct = Math.min(100, (searches / maxSearches) * 100);
  if (searchDone) pct = 100;
  fill.style.width = `${pct}%`;
  if (track) track.setAttribute("aria-valuenow", String(Math.round(pct)));
}

function stopCpuThinking() {
  const counter = document.getElementById("turn-counter");
  const panel = document.getElementById("cpu-thinking");
  const fill = document.getElementById("cpu-thinking-fill");
  const track = document.getElementById("cpu-thinking-track");
  const area = document.getElementById("turn-area");
  if (!counter || !panel || !fill) return;
  cpuThinkingActive = false;
  counter.classList.remove("hidden");
  panel.classList.add("hidden");
  panel.setAttribute("aria-hidden", "true");
  if (area) area.classList.remove("turn-area--thinking");
  fill.style.width = "0%";
  fill.classList.remove("cpu-thinking__fill--indeterminate");
  if (track) track.setAttribute("aria-busy", "false");
}

function finishThinkThen(thinkStart, payload, resolve) {
  const elapsed = performance.now() - thinkStart;
  const wait = Math.max(0, MIN_CPU_THINK_MS - elapsed);
  const finish = () => {
    stopCpuThinking();
    resolve(payload);
  };
  if (wait > 0) {
    setTimeout(finish, wait);
  } else {
    finish();
  }
}

/**
 * Neural MCTS in a Web Worker; falls back to random legal moves on failure.
 * @param {import('./game.js').GameState} gs
 */
function chooseCPUMove(gs) {
  return new Promise((resolve) => {
    const thinkStart = performance.now();
    startCpuThinking();
    function onMsg({ data }) {
      if (data.type === "progress") {
        updateCpuThinking(
          data.searches,
          data.maxSearches,
          Boolean(data.done)
        );
        return;
      }
      if (data.type === "error") {
        cpuWorker.removeEventListener("message", onMsg);
        const fill = document.getElementById("cpu-thinking-fill");
        if (fill && !cpuProgressReceived) {
          fill.classList.add("cpu-thinking__fill--indeterminate");
        }
        finishThinkThen(thinkStart, chooseRandomCpuMove(gs), resolve);
        return;
      }
      if (data.type === "result") {
        cpuWorker.removeEventListener("message", onMsg);
        const fill = document.getElementById("cpu-thinking-fill");
        const track = document.getElementById("cpu-thinking-track");
        if (fill) fill.style.width = "100%";
        if (track) track.setAttribute("aria-valuenow", "100");
        finishThinkThen(thinkStart, data.payload, resolve);
      }
    }
    cpuWorker.addEventListener("message", onMsg);
    cpuWorker.postMessage(gs);
  });
}

function getRulesPages() {
  const refUrl =
    "https://jpneto.github.io/world_abstract_games/corintho.htm";
  return [
    {
      title: "Corintho — overview",
      content: `
        <p><strong>Copyright © 2005 Paolo Scattini, Family Games, Inc.</strong> This page summarizes the rules as described on
        <a href="${refUrl}" target="_blank" rel="noopener noreferrer">Games of Towers — Corintho</a>.</p>
        <h3>Board &amp; pieces</h3>
        <p>Play is on a <strong>4×4</strong> board. A <strong>stack</strong> can contain bases, columns, and capitals.
        A column or capital may sit only on a base or another column. A base cannot be placed on top of anything.
        The <strong>top piece</strong> of a stack defines its type (base, column, or capital).</p>
        <p><strong>Setup:</strong> each player starts with <strong>four bases, four columns, and four capitals</strong> in reserve.</p>
      `,
    },
    {
      title: "Turn &amp; moves",
      content: `
        <p>On your turn you must do exactly one of:</p>
        <ul>
          <li><strong>Drop</strong> one of your pieces from reserve onto a <strong>legal</strong> cell (including stacking a column or capital on a base or column when allowed), or</li>
          <li><strong>Move</strong> a whole stack onto an orthogonally <strong>adjacent</strong> stack so the result is a legal stack.
          Stacks cannot be split.</li>
        </ul>
        <p><strong>Controls:</strong> click a piece in <strong>Your Reserves</strong> or a stack on the board to select it, then click a highlighted cell to place or move. Press <strong>Escape</strong> to clear selection.</p>
        <p><strong>Freeze:</strong> the stack you <em>just created</em> may not be moved on your <em>next</em> turn (shown with an amber tint on the board).</p>
      `,
    },
    {
      title: "Goal",
      content: `
        <p>You win by completing a <strong>stable line of three</strong> stacks in a row (orthogonal), where all three stacks are of the
        <strong>same type</strong> (same top piece).</p>
        <p>A line is <strong>stable</strong> if your opponent cannot break it on their immediate next turn — for example by undoing the threat,
        extending to four in a row, or changing the type of one of the three stacks in the line. (See the worked example on the
        <a href="${refUrl}" target="_blank" rel="noopener noreferrer">rules page</a>.)</p>
        <p>In this web app, line detection matches the Corintho AI engine; when in doubt, trust the game over message.</p>
      `,
    },
    {
      title: "About this site’s CPU",
      content: `
        <p class="rules-bonus"><strong>Bonus — how the computer plays here:</strong> the CPU runs
        <strong>Monte Carlo tree search</strong> guided by a <strong>neural policy and value</strong> model.
        Inference runs in your browser with <strong>ONNX Runtime</strong> inside a <strong>Web Worker</strong> so the page stays responsive.
        If model loading or inference fails, the game falls back to <strong>random legal moves</strong> so you can still play.</p>
      `,
    },
  ];
}

function renderRulesPage() {
  const overlay = document.getElementById("overlay");
  const content = document.getElementById("overlay-content");
  if (!overlay || !content) return;

  const pages = getRulesPages();
  const page = pages[rulesState.page - 1];
  const n = pages.length;

  content.setAttribute("aria-labelledby", "rules-dialog-title");
  content.innerHTML = `
    <h2 class="overlay-title rules-page-title" id="rules-dialog-title">${page.title}</h2>
    <div class="rules-text">${page.content}</div>
    <div class="overlay-btns">
      ${rulesState.page > 1 ? `<button type="button" id="rules-prev" class="btn btn-gray">← Back</button>` : ""}
      <button type="button" id="rules-close" class="btn btn-gray">Close</button>
      ${rulesState.page < n ? `<button type="button" id="rules-next" class="btn btn-indigo">Next →</button>` : ""}
    </div>
    <p class="rules-footer-note">Page ${rulesState.page} of ${n}</p>
  `;

  document.getElementById("rules-close").addEventListener("click", hideRulesOverlay);
  if (rulesState.page > 1) {
    document.getElementById("rules-prev").addEventListener("click", () => {
      rulesState.page--;
      renderRulesPage();
    });
  }
  if (rulesState.page < n) {
    document.getElementById("rules-next").addEventListener("click", () => {
      rulesState.page++;
      renderRulesPage();
    });
  }
}

function showRulesOverlay() {
  const overlay = document.getElementById("overlay");
  if (!overlay) return;
  rulesState.page = 1;
  renderRulesPage();
  overlay.classList.remove("hidden");
  overlay.setAttribute("aria-hidden", "false");
}

function hideRulesOverlay() {
  const overlay = document.getElementById("overlay");
  if (!overlay) return;
  overlay.classList.add("hidden");
  overlay.setAttribute("aria-hidden", "true");
  const content = document.getElementById("overlay-content");
  if (content) content.innerHTML = "";
}

function normalizeMove(moveId) {
  const mtype = moveId >= 48;
  const pieceTypes = ["base", "column", "capital"];

  if (mtype) {
    return {
      type: true,
      pieceType: pieceTypes[Math.floor((moveId - 48) / 16)],
      row: Math.floor((moveId % 16) / 4),
      col: moveId % 4,
    };
  }
  if (moveId < 12) {
    return {
      type: false,
      sourceRow: Math.floor(moveId / 3),
      sourceCol: moveId % 3,
      targetRow: Math.floor(moveId / 3),
      targetCol: moveId % 3 + 1,
    };
  }
  if (moveId < 24) {
    return {
      type: false,
      sourceRow: Math.floor((moveId - 12) / 4),
      sourceCol: moveId % 4,
      targetRow: Math.floor((moveId - 12) / 4) + 1,
      targetCol: moveId % 4,
    };
  }
  if (moveId < 36) {
    return {
      type: false,
      sourceRow: Math.floor((moveId - 24) / 3),
      sourceCol: moveId % 3 + 1,
      targetRow: Math.floor((moveId - 24) / 3),
      targetCol: moveId % 3,
    };
  }
  return {
    type: false,
    sourceRow: Math.floor((moveId - 32) / 4),
    sourceCol: moveId % 4,
    targetRow: Math.floor((moveId - 32) / 4) - 1,
    targetCol: moveId % 4,
  };
}

function doCPUMove(gs, move) {
  if (move === null || gs.turn !== 1) {
    return false;
  }
  let notation;
  if (move.type) {
    notation = formatPlaceNotation(move.pieceType, move.row, move.col);
    gs.placePiece(move.pieceType, move.row, move.col);
  } else {
    notation = formatMoveNotation(
      move.sourceRow,
      move.sourceCol,
      move.targetRow,
      move.targetCol
    );
    gs.movePiece(
      move.sourceRow,
      move.sourceCol,
      move.targetRow,
      move.targetCol
    );
  }
  pushHistory("cpu", notation);
  drawGame(gs);
  return true;
}

function endGame(result, gs) {
  const turnElement = document.getElementById("turn-counter");
  clearSelection();
  if (result === "win") {
    turnElement.textContent = "You won!";
  } else if (result === "draw") {
    turnElement.textContent = "It's a draw!";
  } else {
    turnElement.textContent = "You lost!";
  }
  gs.turn = -1;
  drawGame(gs);
}

export function initCorintho() {
  document.addEventListener("DOMContentLoaded", () => {
    gameState = new GameState();
    moveHistory = [];
    renderHistoryList();
    drawGame(gameState);
    setActivePanel("board");

    document.querySelectorAll(".sidebar-nav-btn[data-panel]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const panel = btn.dataset.panel;
        if (panel) setActivePanel(panel);
      });
    });

    document.getElementById("new-game-btn")?.addEventListener("click", showNewGameOverlay);

    document.getElementById("theme-toggle")?.addEventListener("click", toggleTheme);
    syncThemeToggleUi();

    initQuoteWidget();

    const bank = document.getElementById("piece-bank-player");
    bank?.addEventListener("click", (e) => {
      const btn = e.target.closest("button.piece");
      if (!btn?.id) return;
      if (!PIECE_ORDER.includes(btn.id)) return;
      handleReserveClick(btn.id);
    });

    document.querySelectorAll(".rules-btn").forEach((btn) => {
      btn.addEventListener("click", showRulesOverlay);
    });

    const backdrop = document.getElementById("overlay-backdrop");
    if (backdrop) {
      backdrop.addEventListener("click", hideRulesOverlay);
    }
    document.addEventListener("keydown", (e) => {
      const overlay = document.getElementById("overlay");
      if (e.key === "Escape") {
        if (overlay && !overlay.classList.contains("hidden")) {
          hideRulesOverlay();
        } else {
          clearSelection();
          applySelectionHighlights();
        }
      }
    });
  });
}

initCorintho();

// Accept theme overrides from a parent frame (e.g. personal-website postMessage bridge).
window.addEventListener("message", (e) => {
  if (!e.data || e.data.type !== "theme") return;
  const dark = e.data.value === "dark";
  document.documentElement.classList.toggle("dark", dark);
  try {
    localStorage.setItem(THEME_STORAGE_KEY, dark ? "dark" : "light");
  } catch (_) {
    /* ignore */
  }
  syncThemeToggleUi();
});
