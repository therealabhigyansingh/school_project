// Draws MediaPipe-style 21-landmark hand skeleton on the camera overlay canvas.

const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

export function setupOverlay(videoEl, canvasEl) {
  const sync = () => {
    canvasEl.width = videoEl.clientWidth || 640;
    canvasEl.height = videoEl.clientHeight || 360;
  };
  sync();
  const ro = new ResizeObserver(sync);
  ro.observe(videoEl);
  return canvasEl.getContext("2d");
}

export function drawLandmarks(ctx, landmarks) {
  if (!ctx) return;
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  if (!landmarks) return;

  const w = ctx.canvas.width;
  const h = ctx.canvas.height;

  ctx.strokeStyle = "#3CFF6A";
  ctx.fillStyle = "#3CFF6A";
  ctx.lineWidth = 2;
  ctx.shadowColor = "rgba(60,255,106,0.7)";
  ctx.shadowBlur = 4;

  ctx.beginPath();
  for (const [a, b] of CONNECTIONS) {
    const la = landmarks[a];
    const lb = landmarks[b];
    ctx.moveTo(la[0] * w, la[1] * h);
    ctx.lineTo(lb[0] * w, lb[1] * h);
  }
  ctx.stroke();

  for (const lm of landmarks) {
    ctx.beginPath();
    ctx.arc(lm[0] * w, lm[1] * h, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}
