import { startCamera, startFrameLoop } from "./camera.js";
import { connect } from "./ws.js";
import { setupOverlay, drawLandmarks } from "./overlay.js";

const video = document.getElementById("cam-video");
const overlay = document.getElementById("cam-overlay");
const wordMaskEl = document.getElementById("word-mask");
const wrongListEl = document.getElementById("wrong-list");
const livesDotsEl = document.getElementById("lives-dots");
const currentLetterEl = document.getElementById("current-letter");
const confEl = document.getElementById("conf");
const endDialog = document.getElementById("end-dialog");
const endTitle = document.getElementById("end-title");
const endWord = document.getElementById("end-word");

let ws;
let overlayCtx;
let lastStatus = "playing";

(async function init() {
  try {
    await startCamera(video);
  } catch (err) {
    console.error("camera error", err);
    return;
  }
  overlayCtx = setupOverlay(video, overlay);

  ws = connect("/ws/hangman", { onMessage: handleMessage });
  startFrameLoop({
    video,
    fps: 30,
    onFrame: (blob) => ws.sendBlob(blob),
  });

  document.getElementById("btn-new").addEventListener("click", newGame);
  document.getElementById("btn-again").addEventListener("click", () => {
    endDialog.close();
    newGame();
  });
})();

function newGame() {
  ws?.sendJSON({ action: "new_game" });
  lastStatus = "playing";
}

function handleMessage(msg) {
  drawLandmarks(overlayCtx, msg.landmarks);
  currentLetterEl.textContent = msg.prediction ?? "—";
  confEl.textContent = (msg.confidence ?? 0).toFixed(2);

  const m = msg.mode || {};
  wordMaskEl.textContent = m.word_mask || "";
  wrongListEl.textContent = (m.wrong_guesses || []).join(" ");

  // Lives dots
  const max = m.max_lives ?? 6;
  const left = m.lives_left ?? max;
  livesDotsEl.innerHTML = "";
  for (let i = 0; i < max; i++) {
    const span = document.createElement("span");
    span.className = "dot" + (i < left ? "" : " gone");
    livesDotsEl.appendChild(span);
  }

  // Hangman parts: part 0 is the gallows (always shown), parts 1..6 reveal with wrong guesses
  const wrongCount = (m.wrong_guesses || []).length;
  for (const el of document.querySelectorAll(".hpart")) {
    const part = parseInt(el.dataset.part, 10);
    if (part === 0 || part <= wrongCount) {
      el.classList.add("is-shown");
    } else {
      el.classList.remove("is-shown");
    }
  }

  if (m.status && m.status !== lastStatus) {
    lastStatus = m.status;
    if (m.status === "won" || m.status === "lost") {
      endTitle.textContent = m.status === "won" ? "You won." : "Game over.";
      endWord.textContent = m.word || "";
      if (!endDialog.open) endDialog.showModal();
    }
  }
}
