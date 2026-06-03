import { startCamera, startFrameLoop } from "./camera.js";
import { connect } from "./ws.js";
import { setupOverlay, drawLandmarks } from "./overlay.js";

const video = document.getElementById("cam-video");
const overlay = document.getElementById("cam-overlay");
const letterEl = document.getElementById("letter");
const confEl = document.getElementById("conf");
const confFill = document.getElementById("conf-fill");
const textOut = document.getElementById("text-out");
const handStatus = document.getElementById("hand-status");
const fpsEl = document.getElementById("fps");

let ws;
let overlayCtx;

(async function init() {
  try {
    await startCamera(video);
  } catch (err) {
    handStatus.textContent = `camera error: ${err.message}`;
    return;
  }
  overlayCtx = setupOverlay(video, overlay);

  ws = connect("/ws/decoder", { onMessage: handleMessage });

  startFrameLoop({
    video,
    fps: 30,
    onFrame: (blob) => ws.sendBlob(blob),
    onFps: (n) => (fpsEl.textContent = `${n} fps`),
  });

  document.getElementById("btn-reset").addEventListener("click", () => {
    ws.sendJSON({ action: "reset" });
  });
  document.getElementById("btn-copy").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(textOut.textContent || ""); } catch {}
  });
})();

function handleMessage(msg) {
  drawLandmarks(overlayCtx, msg.landmarks);
  letterEl.textContent = msg.prediction ?? "—";
  const c = msg.confidence ?? 0;
  confEl.textContent = c.toFixed(2);
  confFill.style.width = `${c * 100}%`;

  const m = msg.mode || {};
  textOut.textContent = (m.full_text || "") + (m.current_word || "");
  if (!m.hand_present && m.no_hand_seconds > 0 && m.current_word) {
    handStatus.textContent = `no hand: ${m.no_hand_seconds.toFixed(1)}s / 5.0s`;
  } else {
    handStatus.textContent = "";
  }
}
