// getUserMedia + frame capture loop. Sends mirrored JPEG frames matching the trainer's orientation.

export async function startCamera(videoEl) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: "user" },
    audio: false,
  });
  videoEl.srcObject = stream;
  if (videoEl.readyState < 1) {
    await new Promise((res) => (videoEl.onloadedmetadata = () => res()));
  }
  await videoEl.play();
  return stream;
}

export function startFrameLoop({ video, fps = 30, onFrame, onFps }) {
  const captureCanvas = document.createElement("canvas");
  captureCanvas.width = 320;
  captureCanvas.height = 240;
  const ctx = captureCanvas.getContext("2d");
  const interval = 1000 / fps;

  let last = 0;
  let frames = 0;
  let fpsTick = performance.now();
  let running = true;

  function loop(now) {
    if (!running) return;
    if (now - last >= interval) {
      // Mirror so the backend receives the same orientation the trainer captured.
      ctx.save();
      ctx.translate(captureCanvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
      ctx.restore();
      captureCanvas.toBlob(
        (blob) => blob && onFrame(blob),
        "image/jpeg",
        0.7,
      );
      last = now;
      frames++;
    }
    if (now - fpsTick >= 1000) {
      onFps?.(frames);
      frames = 0;
      fpsTick = now;
    }
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
  return () => { running = false; };
}
