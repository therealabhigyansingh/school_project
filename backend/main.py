import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from inference.predictor import Predictor
from sessions.decoder import DecoderSession
from sessions.practice import PracticeSession
from sessions.hangman import HangmanSession

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT.parent / "frontend"
MODEL_PATH = ROOT / "models" / "asl.pt"

app = FastAPI(title="SignSpeak")
predictor = Predictor(MODEL_PATH)


@app.get("/api/status")
async def status():
    return {"ok": True, "model_loaded": predictor.is_loaded}


async def _run_session(ws: WebSocket, session):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                result = session.process_frame(msg["bytes"])
                await ws.send_json(result)
            elif msg.get("text") is not None:
                try:
                    control = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                result = session.handle_control(control)
                if result is not None:
                    await ws.send_json(result)
    except WebSocketDisconnect:
        pass
    finally:
        session.close()


@app.websocket("/ws/decoder")
async def ws_decoder(ws: WebSocket):
    await _run_session(ws, DecoderSession(predictor))


@app.websocket("/ws/practice")
async def ws_practice(ws: WebSocket):
    await _run_session(ws, PracticeSession(predictor))


@app.websocket("/ws/hangman")
async def ws_hangman(ws: WebSocket):
    await _run_session(ws, HangmanSession(predictor))


# Static frontend mounted last so WS routes take precedence
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
