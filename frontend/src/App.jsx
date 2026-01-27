import React, { useMemo, useRef, useState } from "react";

const API_BASE = "http://localhost:8000";

function formatPct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function drawSpectrogram(canvas, spec) {
  if (!canvas || !spec || spec.length === 0) return;
  const ctx = canvas.getContext("2d");
  const height = spec.length;
  const width = spec[0].length;
  canvas.width = width;
  canvas.height = height;

  const flat = spec.flat();
  const min = Math.min(...flat);
  const max = Math.max(...flat);
  const range = max - min || 1;

  const imageData = ctx.createImageData(width, height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const value = (spec[y][x] - min) / range;
      const intensity = Math.floor(value * 255);
      const idx = (y * width + x) * 4;
      imageData.data[idx + 0] = intensity;
      imageData.data[idx + 1] = intensity;
      imageData.data[idx + 2] = intensity;
      imageData.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(imageData, 0, 0);
}

export default function App() {
  const [file, setFile] = useState(null);
  const [predictResult, setPredictResult] = useState(null);
  const [spectrogram, setSpectrogram] = useState(null);
  const [streaming, setStreaming] = useState(false);
  const [streamStatus, setStreamStatus] = useState("idle");
  const [streamPrediction, setStreamPrediction] = useState(null);
  const [error, setError] = useState(null);

  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const audioCtxRef = useRef(null);
  const processorRef = useRef(null);
  const micSourceRef = useRef(null);

  const topK = useMemo(() => (predictResult ? predictResult.top_k : []), [
    predictResult,
  ]);

  async function handlePredict() {
    if (!file) return;
    setError(null);

    const form = new FormData();
    form.append("file", file);

    const response = await fetch(`${API_BASE}/predict?top_k=5`, {
      method: "POST",
      body: form,
    });
    if (!response.ok) {
      setError(`Predict failed: ${await response.text()}`);
      return;
    }
    const data = await response.json();
    setPredictResult(data);
    if (data.spectrogram?.features) {
      setSpectrogram(data.spectrogram.features);
      requestAnimationFrame(() =>
        drawSpectrogram(canvasRef.current, data.spectrogram.features)
      );
    }
  }

  async function handleSpectrogram() {
    if (!file) return;
    setError(null);

    const form = new FormData();
    form.append("file", file);

    const response = await fetch(`${API_BASE}/spectrogram`, {
      method: "POST",
      body: form,
    });
    if (!response.ok) {
      setError(`Spectrogram failed: ${await response.text()}`);
      return;
    }
    const data = await response.json();
    setSpectrogram(data.features);
    requestAnimationFrame(() =>
      drawSpectrogram(canvasRef.current, data.features)
    );
  }

  async function startStream() {
    setError(null);
    setStreamStatus("connecting");
    const ws = new WebSocket("ws://localhost:8000/ws/predict");
    wsRef.current = ws;

    ws.onopen = async () => {
      setStreamStatus("streaming");
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      audioCtxRef.current = audioCtx;

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const source = audioCtx.createMediaStreamSource(stream);
      micSourceRef.current = source;

      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      ws.send(
        JSON.stringify({
          sample_rate: audioCtx.sampleRate,
          duration: 2.0,
          n_mels: 64,
        })
      );

      processor.onaudioprocess = (event) => {
        const input = event.inputBuffer.getChannelData(0);
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(input.buffer);
        }
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);
      setStreaming(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.error) {
          setError(data.error);
          return;
        }
        setStreamPrediction(data);
      } catch (err) {
        setError(`Stream parse error: ${err}`);
      }
    };

    ws.onerror = () => {
      setStreamStatus("error");
    };

    ws.onclose = () => {
      setStreamStatus("idle");
      setStreaming(false);
    };
  }

  function stopStream() {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (micSourceRef.current) {
      micSourceRef.current.disconnect();
      micSourceRef.current = null;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    setStreaming(false);
    setStreamStatus("idle");
  }

  return (
    <div className="app">
      <header>
        <h1>Environmental Audio CNN</h1>
        <p>Upload audio, see predictions, and view spectrograms.</p>
      </header>

      <section className="panel">
        <h2>Upload Audio</h2>
        <input
          type="file"
          accept="audio/*"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
        <div className="actions">
          <button onClick={handlePredict} disabled={!file}>
            Predict
          </button>
          <button onClick={handleSpectrogram} disabled={!file}>
            Spectrogram
          </button>
        </div>
      </section>

      <section className="panel">
        <h2>Predictions</h2>
        {predictResult ? (
          <div>
            <div className="primary">
              <strong>{predictResult.top_prediction.label}</strong>{" "}
              <span>{formatPct(predictResult.top_prediction.confidence)}</span>
            </div>
            <ul className="list">
              {topK.map((item) => (
                <li key={item.label}>
                  <span>{item.label}</span>
                  <span>{formatPct(item.confidence)}</span>
                </li>
              ))}
            </ul>
          </div>
        ) : (
          <p>No predictions yet.</p>
        )}
      </section>

      <section className="panel">
        <h2>Spectrogram</h2>
        {spectrogram ? (
          <canvas ref={canvasRef} className="spectrogram" />
        ) : (
          <p>No spectrogram yet.</p>
        )}
      </section>

      <section className="panel">
        <h2>Realtime Mic</h2>
        <p>Status: {streamStatus}</p>
        <div className="actions">
          <button onClick={startStream} disabled={streaming}>
            Start Streaming
          </button>
          <button onClick={stopStream} disabled={!streaming}>
            Stop
          </button>
        </div>
        {streamPrediction ? (
          <div className="primary">
            <strong>{streamPrediction.label}</strong>{" "}
            <span>{formatPct(streamPrediction.confidence)}</span>
          </div>
        ) : (
          <p>No live prediction yet.</p>
        )}
      </section>

      {error ? <div className="error">{error}</div> : null}
    </div>
  );
}
