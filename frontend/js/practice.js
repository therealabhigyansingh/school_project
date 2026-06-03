import { startCamera, startFrameLoop } from "./camera.js";
import { connect } from "./ws.js";
import { setupOverlay, drawLandmarks } from "./overlay.js";

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const RING_CIRC = 339.292;

const picker = document.getElementById("letter-picker");
const view = document.getElementById("practice-view");
const grid = document.getElementById("letter-grid");
const targetLetterEl = document.getElementById("target-letter");
const refImg = document.getElementById("reference-img");
const ringFg = document.getElementById("ring-fg");
const holdText = document.getElementById("hold-text");
const currentLetterEl = document.getElementById("current-letter");
const confEl = document.getElementById("conf");
const feedbackEl = document.getElementById("feedback");
const successDialog = document.getElementById("success-dialog");
const successLetterEl = document.getElementById("success-letter");

let ws;
let overlayCtx;
let videoStarted = false;
let currentTarget = null;

for (const ltr of LETTERS) {
  const btn = document.createElement("button");
  btn.className = "tile";
  btn.textContent = ltr;
  btn.dataset.letter = ltr;
  btn.addEventListener("click", () => startLetter(ltr));
  grid.appendChild(btn);
}

document.getElementById("btn-back").addEventListener("click", exitPractice);
document.getElementById("btn-exit").addEventListener("click", () => {
  successDialog.close();
  exitPractice();
});
document.getElementById("btn-next").addEventListener("click", () => {
  successDialog.close();
  nextLetter();
});

refImg.addEventListener("load", () => refImg.classList.add("is-loaded"));
refImg.addEventListener("error", () => refImg.classList.remove("is-loaded"));

async function startLetter(letter) {
  currentTarget = letter;
  targetLetterEl.textContent = letter;
  refImg.classList.remove("is-loaded");
  refImg.src = `assets/reference/${letter}_test.jpg`;
  picker.hidden = true;
  view.hidden = false;

  if (!videoStarted) {
    const video = document.getElementById("cam-video");
    const overlay = document.getElementById("cam-overlay");
    try {
      await startCamera(video);
    } catch (err) {
      console.error("camera error", err);
      return;
    }
    overlayCtx = setupOverlay(video, overlay);
    ws = connect("/ws/practice", { onMessage: handleMessage });
    startFrameLoop({
      video,
      fps: 30,
      onFrame: (blob) => ws.sendBlob(blob),
    });
    videoStarted = true;
  }
  ws?.sendJSON({ action: "start", letter });
  ringFg.style.strokeDashoffset = RING_CIRC;
  holdText.textContent = "0.0 / 5.0s";
}

function nextLetter() {
  const idx = LETTERS.indexOf(currentTarget);
  startLetter(LETTERS[(idx + 1) % LETTERS.length]);
}

function exitPractice() {
  picker.hidden = false;
  view.hidden = true;
  ws?.sendJSON({ action: "reset" });
  ringFg.style.strokeDashoffset = RING_CIRC;
  feedbackEl.textContent = "";
  feedbackEl.classList.remove("is-correct", "is-incorrect");
}

function handleMessage(msg) {
  drawLandmarks(overlayCtx, msg.landmarks);
  currentLetterEl.textContent = msg.prediction ?? "—";
  const conf = msg.confidence ?? 0;
  confEl.textContent = conf.toFixed(2);

  const progress = msg.mode?.hold_progress ?? 0;
  ringFg.style.strokeDashoffset = RING_CIRC * (1 - progress);
  holdText.textContent = `${(progress * 5).toFixed(1)} / 5.0s`;

  const isCorrect = msg.prediction === currentTarget && conf > 0.60;
  feedbackEl.textContent = isCorrect ? "Yes, That's it!" : "Not Quite!";
  feedbackEl.classList.toggle("is-correct", isCorrect);
  feedbackEl.classList.toggle("is-incorrect", !isCorrect);

  if (msg.mode?.success && !successDialog.open) {
    successLetterEl.textContent = currentTarget;
    successDialog.showModal();
  }
}
