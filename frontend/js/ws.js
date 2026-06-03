// Thin WebSocket helper that wires the footer status indicator.

export function connect(path, { onOpen, onMessage, onClose, onError } = {}) {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${proto}//${location.host}${path}`;
  const ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";

  // Queue JSON messages sent before the socket finishes its handshake.
  // Frames (binary) aren't queued — they stream continuously, dropping a few is fine.
  let pendingJSON = [];

  setStatus("connecting");
  ws.onopen = () => {
    setStatus("connected");
    for (const m of pendingJSON) ws.send(m);
    pendingJSON = null;
    onOpen?.();
  };
  ws.onmessage = (e) => {
    try { onMessage?.(JSON.parse(e.data)); } catch { /* ignore */ }
  };
  ws.onclose = () => { setStatus("idle"); onClose?.(); };
  ws.onerror = () => { setStatus("error"); onError?.(); };

  return {
    sendBlob: (blob) => {
      if (ws.readyState !== 1) return;
      blob.arrayBuffer().then((buf) => {
        if (ws.readyState === 1) ws.send(buf);
      });
    },
    sendJSON: (obj) => {
      const msg = JSON.stringify(obj);
      if (ws.readyState === 1) {
        ws.send(msg);
      } else if (pendingJSON) {
        pendingJSON.push(msg);
      }
    },
    close: () => ws.close(),
  };
}

function setStatus(state) {
  const dot = document.getElementById("ws-status-dot");
  const text = document.getElementById("ws-status-text");
  if (!dot || !text) return;
  dot.className = "status__dot";
  if (state === "connected") {
    dot.classList.add("is-connected");
    text.textContent = "connected";
  } else if (state === "error") {
    dot.classList.add("is-error");
    text.textContent = "error";
  } else if (state === "connecting") {
    text.textContent = "connecting";
  } else {
    text.textContent = "idle";
  }
}
