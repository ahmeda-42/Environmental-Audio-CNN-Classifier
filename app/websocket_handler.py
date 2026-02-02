import logging
from functools import lru_cache
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import WebSocket, WebSocketDisconnect
from app.schemas import StreamConfig
from model.load_model import MODEL_PATH, load_model
from model.predict import compute_spectrogram_item
from model.dataset import load_label_mapping


@lru_cache(maxsize=1)
def _get_stream_resources():
    label_to_index = load_label_mapping(MODEL_PATH + ".labels.json")
    index_to_label = {v: k for k, v in label_to_index.items()}
    model, device = load_model(num_classes=len(label_to_index))
    return index_to_label, model, device

async def handle_websocket_predict(websocket: WebSocket):
    logger = logging.getLogger("uvicorn.error")
    await websocket.accept()
    logger.info("WebSocket connected.")

    # First message should be JSON config for the stream
    config_raw = await websocket.receive_text()
    try:
        config = StreamConfig.model_validate_json(config_raw)
        sample_rate = config.sample_rate
        duration = config.duration
        n_mels = config.n_mels
        logger.info("WebSocket config: sr=%s duration=%s n_mels=%s", sample_rate, duration, n_mels)
    except Exception as exc:
        logger.warning("WebSocket config invalid: %s", exc)
        await websocket.send_json({"error": f"invalid config: {exc}"})
        await websocket.close()
        return

    index_to_label, model, device = _get_stream_resources()

    target_len = int(sample_rate * duration)
    buffer = np.zeros(0, dtype=np.float32)

    # Client sends raw float32 mono PCM in binary messages
    while True:
        try:
            message = await websocket.receive()
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected.")
            break
        if message.get("type") == "websocket.disconnect":
            break
        if "bytes" in message:
            try:
                if len(message["bytes"]) % 4 != 0:
                    await websocket.send_json({"error": "invalid audio chunk size"})
                    continue
                chunk = np.frombuffer(message["bytes"], dtype=np.float32)
                if chunk.size == 0:
                    continue
                buffer = np.concatenate([buffer, chunk])

                # Keep only the most recent window
                if buffer.size > target_len * 2:
                    buffer = buffer[-target_len * 2 :]

                if buffer.size >= target_len:
                    window = buffer[-target_len:]
                    feat, window_item = compute_spectrogram_item(
                        sample_rate=sample_rate,
                        duration=duration,
                        n_mels=n_mels,
                        y=window,
                        sr=sample_rate,
                        include_image=False,
                        include_features=True,
                    )
                    window_item = {
                        **window_item,
                        "window_start": 0.0,
                        "window_end": float(duration),
                    }
                    x = torch.tensor(feat).unsqueeze(0).unsqueeze(0)

                    with torch.no_grad():
                        logits = model(x.to(device))
                        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

                    top_k = 4
                    top_k = max(1, min(top_k, len(probs)))
                    top_indices = np.argsort(probs)[-top_k:][::-1]
                    top_predictions = [
                        {"label": index_to_label[i], "confidence": float(probs[i])}
                        for i in top_indices
                    ]
                    await websocket.send_json(
                        {
                            "top_prediction": top_predictions[0],
                            "top_k": top_predictions,
                            "spectrogram": window_item,
                            "spectrograms": [window_item],
                        }
                    )
            except Exception as exc:
                logger.warning("WebSocket stream error: %s", exc)
                await websocket.send_json({"error": "stream processing failed"})
        elif "text" in message:
            # Allow client to reset the buffer
            if message["text"].strip().lower() == "reset":
                buffer = np.zeros(0, dtype=np.float32)
