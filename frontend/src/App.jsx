import React, { useEffect, useMemo, useRef, useState } from "react";
import linkedinIcon from "./assets/linkedin-112.svg";
import githubIcon from "./assets/github-logo-6528.svg";
import gmailIcon from "./assets/google-gmail-black-24179.svg";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

function toWebSocketUrl(baseUrl) {
  if (baseUrl.startsWith("https://")) return baseUrl.replace("https://", "wss://");
  if (baseUrl.startsWith("http://")) return baseUrl.replace("http://", "ws://");
  return baseUrl;
}

function formatPct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatLabel(label) {
  if (!label) return "";
  return label
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function formatHz(value) {
  if (!Number.isFinite(value)) return "";
  if (value >= 1000) return `${(value / 1000).toFixed(1)} kHz`;
  return `${Math.round(value)} Hz`;
}

function formatSeconds(value) {
  if (!Number.isFinite(value)) return "";
  return `${value.toFixed(1)} s`;
}

function melToHz(mel) {
  return 700 * (10 ** (mel / 2595) - 1);
}

function hzToMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}

function computeTickValues(spec, meta) {
  const specHeight = spec.length;
  const specWidth = spec[0]?.length || 0;
  const numTicks = 5;

  const sampleRate = meta.sample_rate || 22050;
  const hopLength = meta.hop_length || 512;
  const nMels = meta.n_mels || specHeight;

  const duration = specWidth > 1 ? ((specWidth - 1) * hopLength) / sampleRate : 0;
  let timeTicks = meta.time_ticks || [];
  if (timeTicks.length === 0) {
    timeTicks = Array.from({ length: numTicks }, (_, i) => (i * duration) / (numTicks - 1 || 1));
  }
  const roundedDuration = Math.round(duration * 10) / 10;
  if (timeTicks.length > 0) {
    timeTicks = timeTicks.map((t) => Math.round(t * 10) / 10);
    timeTicks[timeTicks.length - 1] = roundedDuration;
  }

  let freqTicks = meta.freq_ticks || [];
  if (freqTicks.length === 0) {
    const melMin = hzToMel(0);
    const melMax = hzToMel(sampleRate / 2);
    freqTicks = Array.from({ length: numTicks }, (_, i) =>
      melToHz(melMin + (i * (melMax - melMin)) / (numTicks - 1 || 1))
    );
  }

  let dbTicks = meta.db_ticks || [];
  if (dbTicks.length === 0) {
    const flat = spec.flat();
    const min = Math.min(...flat);
    const max = Math.max(...flat);
    const mid = (min + max) / 2;
    dbTicks = [max, mid, min];
  }

  return { timeTicks, freqTicks, dbTicks };
}

function drawSpectrogram(canvas, spec, meta) {
  if (!canvas || !spec || spec.length === 0 || !meta) return;
  const ctx = canvas.getContext("2d");
  const specHeight = spec.length;
  const specWidth = spec[0].length;
  const ticks = computeTickValues(spec, meta);

  const margin = { left: 90, right: 110, top: 30, bottom: 55 };
  const imageWidth = 720;
  const imageHeight = 320;
  const colorbarWidth = 16;
  const colorbarGap = 50;

  canvas.width = margin.left + imageWidth + colorbarGap + colorbarWidth + margin.right;
  canvas.height = margin.top + imageHeight + margin.bottom;
  canvas.style.width = `${canvas.width}px`;
  canvas.style.height = `${canvas.height}px`;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Build spectrogram bitmap (grayscale)
  const flat = spec.flat();
  const min = Math.min(...flat);
  const max = Math.max(...flat);
  const range = max - min || 1;
  const offscreen = document.createElement("canvas");
  offscreen.width = specWidth;
  offscreen.height = specHeight;
  const offCtx = offscreen.getContext("2d");
  const imageData = offCtx.createImageData(specWidth, specHeight);
  for (let y = 0; y < specHeight; y += 1) {
    const srcY = specHeight - 1 - y;
    for (let x = 0; x < specWidth; x += 1) {
      const value = (spec[srcY][x] - min) / range;
      const intensity = Math.floor(value * 255);
      const idx = (y * specWidth + x) * 4;
      imageData.data[idx + 0] = intensity;
      imageData.data[idx + 1] = intensity;
      imageData.data[idx + 2] = intensity;
      imageData.data[idx + 3] = 255;
    }
  }
  offCtx.putImageData(imageData, 0, 0);
  ctx.drawImage(offscreen, margin.left, margin.top, imageWidth, imageHeight);

  // Axes
  ctx.strokeStyle = "#0f172a";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + imageHeight);
  ctx.lineTo(margin.left + imageWidth, margin.top + imageHeight);
  ctx.stroke();

  // Y ticks (frequency)
  const yTicks = ticks.freqTicks || [];
  ctx.fillStyle = "#0f172a";
  ctx.font = "14px Inter, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  yTicks.forEach((tick, i) => {
    const y = margin.top + (imageHeight * i) / (yTicks.length - 1 || 1);
    ctx.beginPath();
    ctx.moveTo(margin.left - 6, y);
    ctx.lineTo(margin.left, y);
    ctx.stroke();
    const label = formatHz(tick);
    ctx.fillText(label, margin.left - 10, y);
  });

  // X ticks (time)
  const xTicks = ticks.timeTicks || [];
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  xTicks.forEach((tick, i) => {
    const x = margin.left + (imageWidth * i) / (xTicks.length - 1 || 1);
    ctx.beginPath();
    ctx.moveTo(x, margin.top + imageHeight);
    ctx.lineTo(x, margin.top + imageHeight + 6);
    ctx.stroke();
    const label = formatSeconds(tick);
    ctx.fillText(label, x, margin.top + imageHeight + 22);
  });

  // Axis labels
  ctx.fillStyle = "#0f172a";
  ctx.font = "15px Inter, sans-serif";
  ctx.save();
  ctx.translate(20, margin.top + imageHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Frequency", 0, 0);
  ctx.restore();
  ctx.fillText(
    "Time",
    margin.left + imageWidth / 2 - 24,
    margin.top + imageHeight + 44
  );

  // Colorbar
  const cbX = margin.left + imageWidth + colorbarGap;
  const cbY = margin.top;
  const cbHeight = imageHeight;
  const gradient = ctx.createLinearGradient(0, cbY, 0, cbY + cbHeight);
  gradient.addColorStop(0, "#ffffff");
  gradient.addColorStop(1, "#111827");
  ctx.fillStyle = gradient;
  ctx.fillRect(cbX, cbY, colorbarWidth, cbHeight);

  const dbTicks = ticks.dbTicks || [];
  ctx.fillStyle = "#0f172a";
  ctx.font = "14px Inter, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  dbTicks.forEach((tick, i) => {
    const y = cbY + (cbHeight * i) / (dbTicks.length - 1 || 1);
    ctx.beginPath();
    ctx.moveTo(cbX + colorbarWidth, y);
    ctx.lineTo(cbX + colorbarWidth + 6, y);
    ctx.stroke();
    ctx.fillText(`${tick.toFixed(1)} dB`, cbX + colorbarWidth + 10, y);
  });
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  ctx.fillText("Intensity", cbX - 6, cbY + cbHeight + 20);
}

export default function App() {
  const [file, setFile] = useState(null);
  const [inputMode, setInputMode] = useState(null);
  const [audioUrl, setAudioUrl] = useState("");
  const [predictResult, setPredictResult] = useState(null);
  const [spectrogram, setSpectrogram] = useState(null);
  const [spectrogramMeta, setSpectrogramMeta] = useState(null);
  const [streaming, setStreaming] = useState(false);
  const [streamStatus, setStreamStatus] = useState("idle");
  const [streamPrediction, setStreamPrediction] = useState(null);
  const [error, setError] = useState(null);

  const canvasRef = useRef(null);
  const predictionsRef = useRef(null);
  const FIXED_PANEL_HEIGHT = 265;
  const [fixedPanelHeight] = useState(FIXED_PANEL_HEIGHT);
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

    try {
      const response = await fetch(`${API_BASE}/predict?top_k=4`, {
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
        setSpectrogramMeta(data.spectrogram);
      }
    } catch (err) {
      setError(`Predict failed: ${err}`);
    }
  }

  async function handleSpectrogram() {
    if (!file) return;
    setError(null);

    const form = new FormData();
    form.append("file", file);

    try {
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
      setSpectrogramMeta(data);
    } catch (err) {
      setError(`Spectrogram failed: ${err}`);
    }
  }

  async function startStream() {
    setError(null);
    setStreamStatus("connecting");
    const ws = new WebSocket(`${toWebSocketUrl(API_BASE)}/ws/predict`);
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

  useEffect(() => {
    if (spectrogram && spectrogramMeta) {
      requestAnimationFrame(() =>
        drawSpectrogram(canvasRef.current, spectrogram, spectrogramMeta)
      );
    }
  }, [spectrogram, spectrogramMeta]);

  useEffect(() => {
    if (!file) {
      setAudioUrl("");
      return;
    }
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    if (predictionsRef.current) {
      predictionsRef.current.style.height = `${FIXED_PANEL_HEIGHT}px`;
    }
  }, [FIXED_PANEL_HEIGHT]);

  return (
    <div className="app">
      <header>
        <h1>Environmental Audio CNN Classifier</h1>
        <p>Upload audio, see predictions, and view spectrograms.</p>
      </header>

      <div className="panel-grid">
        <section
          className="panel audio-panel"
          style={fixedPanelHeight ? { height: fixedPanelHeight } : undefined}
        >
          <div className="panel-header">
            <h2>
              {inputMode
                ? inputMode === "upload"
                  ? "Upload Audio"
                  : "Realtime Mic"
                : "Audio Input"}
            </h2>
          </div>
          {!inputMode ? (
            <div className="mode-grid">
              <button onClick={() => setInputMode("upload")}>
                <img
                  className="mode-icon"
                  src="https://images.icon-icons.com/1875/PNG/512/fileupload_120150.png"
                  alt="Upload"
                />
                Upload Audio
              </button>
              <button onClick={() => setInputMode("mic")}>
                <img
                  className="mode-icon"
                  src="https://cdn-icons-png.flaticon.com/512/1082/1082810.png"
                  alt="Mic"
                />
                Realtime Mic
              </button>
            </div>
          ) : inputMode === "upload" ? (
            <div className="input-body">
              <div className="file-row">
                <label className="file-picker">
                  Choose File
                  <input
                    className="file-input"
                    type="file"
                    accept="audio/*"
                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                  />
                </label>
                <span className="file-name">
                  {file ? file.name : "No file chosen"}
                </span>
              </div>
              {audioUrl ? (
                <audio className="audio-player" controls src={audioUrl} />
              ) : null}
              <div className="input-footer">
                <button className="back-button action-button" onClick={() => setInputMode(null)}>
                  Back
                </button>
                <div className="actions">
                  <button className="action-button" onClick={handlePredict} disabled={!file}>
                    Predict
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="input-body">
              <p className="muted">Status: {streamStatus}</p>
              {streamPrediction ? (
                <div className="primary">
                  <div className="prediction-row">
                    <strong>{formatLabel(streamPrediction.label)}</strong>
                    <span>{formatPct(streamPrediction.confidence)}</span>
                  </div>
                  <div className="confidence-bar">
                    <div
                      className="confidence-fill"
                      style={{
                        width: `${streamPrediction.confidence * 100}%`,
                      }}
                    />
                  </div>
                </div>
              ) : null}
              <div className="input-footer">
                <button className="back-button action-button" onClick={() => setInputMode(null)}>
                  Back
                </button>
                <div className="actions">
                  <button className="action-button" onClick={streaming ? stopStream : startStream}>
                    {streaming ? "Stop Streaming" : "Start Streaming"}
                  </button>
                </div>
              </div>
            </div>
          )}
        </section>

        <section
          className="panel predictions-panel"
          ref={predictionsRef}
          style={fixedPanelHeight ? { height: fixedPanelHeight } : undefined}
        >
          <h2>Top Predictions</h2>
          {predictResult ? (
            <div>
              <div className="primary">
                <div className="prediction-row">
                  <strong>{formatLabel(predictResult.top_prediction.label)}</strong>
                  <span>{formatPct(predictResult.top_prediction.confidence)}</span>
                </div>
                <div className="confidence-bar">
                  <div
                    className="confidence-fill"
                    style={{
                      width: `${predictResult.top_prediction.confidence * 100}%`,
                    }}
                  />
                </div>
              </div>
              <ul className="list">
                {topK
                  .filter(
                    (item) => item.label !== predictResult.top_prediction.label
                  )
                  .map((item) => (
                    <li key={item.label}>
                      <div className="prediction-row">
                        <span>{formatLabel(item.label)}</span>
                        <span>{formatPct(item.confidence)}</span>
                      </div>
                      <div className="confidence-bar">
                        <div
                          className="confidence-fill"
                          style={{ width: `${item.confidence * 100}%` }}
                        />
                      </div>
                    </li>
                  ))}
              </ul>
            </div>
          ) : (
            <p>No predictions yet.</p>
          )}
        </section>
      </div>

      <section className="panel">
        <h2>Log-Mel Spectrogram</h2>
        {spectrogram ? (
          <div className="spectrogram-frame">
            <div className="spec-canvas">
              <canvas ref={canvasRef} className="spectrogram" />
            </div>
          </div>
        ) : (
          <p>No spectrogram yet.</p>
        )}
      </section>

      {error ? <div className="error">{error}</div> : null}

      <footer className="footer">
        <div className="footer-title">Built by Ahmed Alhakem</div>
        <div className="footer-links">
          <a
            href="https://www.linkedin.com/in/ahmedalhakem/"
            target="_blank"
            rel="noreferrer"
            aria-label="LinkedIn"
            className="icon-only"
          >
            <img
              className="footer-icon"
              src={linkedinIcon}
              alt="LinkedIn"
            />
          </a>
          <a
            href="mailto:ahmedalhakem42@gmail.com"
            target="_blank"
            rel="noreferrer"
            aria-label="Email"
            className="icon-only"
          >
            <img
              className="footer-icon"
              src={gmailIcon}
              alt="Email"
            />
          </a>
          <a
            href="https://github.com/ahmeda-42"
            target="_blank"
            rel="noreferrer"
            aria-label="GitHub"
            className="icon-only"
          >
            <img
              className="footer-icon"
              src={githubIcon}
              alt="GitHub"
            />
          </a>
        </div>
      </footer>
    </div>
  );
}
