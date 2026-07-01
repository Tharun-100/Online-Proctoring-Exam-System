import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import xgboost as xgb
from flask import Flask, jsonify, redirect, render_template_string, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from src.feature_extraction2 import FeatureExtractor

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(APP_ROOT, "models", "xgboost_window_model.json")
META_PATH = os.path.join(APP_ROOT, "models", "xgboost_window_model_meta.json")

UPLOAD_DIR = os.path.join(APP_ROOT, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

WINDOW_SECONDS = 1.0
SUSPICIOUS_RATIO_THRESHOLD = 0.8  # same threshold used in src/data_video_creation.py


@dataclass
class WindowResult:
    window_start_sec: float
    window_end_sec: float
    features: Dict[str, float]
    suspicious_types: List[str]
    probability: float


_RESOURCES: Dict[str, Any] = {}


def _load_resources() -> Dict[str, Any]:
    global _RESOURCES
    if _RESOURCES:
        return _RESOURCES

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"XGBoost model not found: {MODEL_PATH}")
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"XGBoost meta not found: {META_PATH}")

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_columns = list(meta.get("feature_columns", []))
    if not feature_columns:
        raise ValueError("No feature_columns found in model meta JSON.")

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    extractor = FeatureExtractor()

    _RESOURCES = {
        "meta": meta,
        "feature_columns": feature_columns,
        "decision_threshold": float(meta.get("decision_threshold", 0.5)),
        "model": model,
        "extractor": extractor,
    }
    return _RESOURCES


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _create_window_features(window_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Match the feature engineering from src/data_video_creation.py#create_window_row
    (excluding label, video_id, and source_folder).
    """
    if not window_rows:
        return {
            "phone_present_ratio": 0.0,
            "max_phone_conf": 0.0,
            "avg_num_faces": 0.0,
            "multiple_faces_ratio": 0.0,
            "head_pose_away_ratio": 0.0,
            "avg_head_pitch": 0.0,
            "avg_head_yaw": 0.0,
            "avg_head_roll": 0.0,
        }

    phone_present = np.array([int(r.get("phone_present", 0) or 0) for r in window_rows], dtype=np.float32)
    phone_conf = np.array([float(r.get("phone_conf", 0.0) or 0.0) for r in window_rows], dtype=np.float32)
    num_faces = np.array([float(r.get("num_faces", 0) or 0) for r in window_rows], dtype=np.float32)
    multiple_faces = np.array([int(r.get("multiple_faces", 0) or 0) for r in window_rows], dtype=np.float32)
    head_pose_away = np.array([int(r.get("head_pose_away", 0) or 0) for r in window_rows], dtype=np.float32)
    head_pitch = np.array([float(r.get("head_pitch", 0.0) or 0.0) for r in window_rows], dtype=np.float32)
    head_yaw = np.array([float(r.get("head_yaw", 0.0) or 0.0) for r in window_rows], dtype=np.float32)
    head_roll = np.array([float(r.get("head_roll", 0.0) or 0.0) for r in window_rows], dtype=np.float32)
    return {
        "phone_present_ratio": float(phone_present.mean()) if phone_present.size else 0.0,
        "max_phone_conf": float(phone_conf.max()) if phone_conf.size else 0.0,
        "avg_num_faces": float(num_faces.mean()) if num_faces.size else 0.0,
        "multiple_faces_ratio": float(multiple_faces.mean()) if multiple_faces.size else 0.0,
        "head_pose_away_ratio": float(head_pose_away.mean()) if head_pose_away.size else 0.0,
        "avg_head_pitch": float(head_pitch.mean()) if head_pitch.size else 0.0,
        "avg_head_yaw": float(head_yaw.mean()) if head_yaw.size else 0.0,
        "avg_head_roll": float(head_roll.mean()) if head_roll.size else 0.0,
    }


def _window_suspicious_types(window_features: Dict[str, float]) -> List[str]:
    types: List[str] = []
    if float(window_features.get("phone_present_ratio", 0.0)) >= SUSPICIOUS_RATIO_THRESHOLD:
        types.append("Mobile phone usage")
    if float(window_features.get("multiple_faces_ratio", 0.0)) >= SUSPICIOUS_RATIO_THRESHOLD:
        types.append("Multiple faces detected")
    if float(window_features.get("head_pose_away_ratio", 0.0)) >= SUSPICIOUS_RATIO_THRESHOLD:
        types.append("Head pose away from screen")
    return types


def _dominant_suspicious_signal(window_features: Dict[str, float]) -> Tuple[str, float]:
    candidates = {
        "Mobile phone usage": float(window_features.get("phone_present_ratio", 0.0) or 0.0),
        "Multiple faces detected": float(window_features.get("multiple_faces_ratio", 0.0) or 0.0),
        "Head pose away from screen": float(window_features.get("head_pose_away_ratio", 0.0) or 0.0),
    }
    best_type = max(candidates, key=candidates.get)
    return best_type, float(candidates[best_type])


def _streaks_from_flags(
    window_rows: List[Dict[str, Any]], flags: List[bool]
) -> Tuple[float, float, List[Dict[str, float]]]:
    total_seconds = float(sum(1.0 for f in flags if f) * WINDOW_SECONDS)

    streaks: List[Dict[str, float]] = []
    longest = 0.0
    i = 0
    n = len(flags)
    while i < n:
        if not flags[i]:
            i += 1
            continue
        start = float(window_rows[i].get("window_start_sec", i * WINDOW_SECONDS) or (i * WINDOW_SECONDS))
        j = i
        while j < n and flags[j]:
            j += 1
        end = float(window_rows[j - 1].get("window_end_sec", j * WINDOW_SECONDS) or (j * WINDOW_SECONDS))
        seconds = float(end - start)
        longest = max(longest, seconds)
        streaks.append({"start_sec": start, "end_sec": end, "seconds": seconds})
        i = j

    return total_seconds, float(longest), streaks


def _compute_problem_summary(
    window_rows: List[Dict[str, Any]],
    *,
    ratio_threshold: float,
    prolonged_seconds: float,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    if not window_rows:
        return None, []

    signal_defs = [
        ("Mobile phone usage", "phone_present_ratio"),
        ("Multiple faces detected", "multiple_faces_ratio"),
        ("Head pose away from screen", "head_pose_away_ratio"),
    ]

    summaries: List[Dict[str, Any]] = []
    best_exact: Optional[Dict[str, Any]] = None

    for signal_name, key in signal_defs:
        flags = [float(w.get(key, 0.0) or 0.0) >= float(ratio_threshold) for w in window_rows]
        total_s, longest_s, streaks = _streaks_from_flags(window_rows, flags)
        is_prolonged = bool(longest_s >= float(prolonged_seconds))
        summary = {
            "type": signal_name,
            "ratio_threshold": float(ratio_threshold),
            "prolonged_seconds": float(prolonged_seconds),
            "total_seconds": float(total_s),
            "longest_streak_seconds": float(longest_s),
            "is_prolonged": is_prolonged,
            "streaks": streaks,
        }
        summaries.append(summary)

        if longest_s <= 0:
            continue
        if best_exact is None:
            best_exact = summary
            continue
        # Prefer prolonged, then longest streak, then total time.
        if summary["is_prolonged"] and not best_exact["is_prolonged"]:
            best_exact = summary
        elif summary["is_prolonged"] == best_exact["is_prolonged"]:
            if longest_s > float(best_exact["longest_streak_seconds"]):
                best_exact = summary
            elif longest_s == float(best_exact["longest_streak_seconds"]) and total_s > float(best_exact["total_seconds"]):
                best_exact = summary

    return best_exact, summaries


def _extract_windows_from_video(video_path: str,*,fps_sample: Optional[float] = None,max_seconds: Optional[float] = None,) -> Tuple[List[Dict[str, Any]], float]:
    """
    Returns: (window_feature_rows, video_fps)
    Each row has window_start_sec, window_end_sec, and the engineered window features.
    """
    resources = _load_resources()
    extractor: FeatureExtractor = resources["extractor"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not fps or fps <= 0:
        fps = 30.0

    target_fps = None
    if fps_sample is not None:
        try:
            fps_sample = float(fps_sample)
        except Exception:
            fps_sample = None
    if fps_sample is not None and fps_sample > 0:
        target_fps = min(fps, fps_sample)

    sample_period_s = (1.0 / target_fps) if target_fps else None
    next_sample_ts = 0.0

    frame_id = 0
    current_window_start = 0.0
    current_window_end = WINDOW_SECONDS
    current_window_rows: List[Dict[str, Any]] = []
    window_rows: List[Dict[str, Any]] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = frame_id / fps
        frame_id += 1

        if max_seconds is not None and timestamp_sec >= float(max_seconds):
            break

        if sample_period_s is not None and timestamp_sec + 1e-9 < next_sample_ts:
            continue

        features = extractor.extract_features(frame)
        num_faces = int(features.get("number_of_people", 0) or 0)

        row = {
            "timestamp_sec": float(timestamp_sec),
            "phone_present": int(features.get("phone_present", 0) or 0),
            "phone_conf": float(features.get("phone_conf", 0.0) or 0.0),
            "num_faces": int(num_faces),
            "multiple_faces": int(num_faces > 1),
            "head_pose_away": int(features.get("head_pose", 0) or 0),
            "head_pitch": float(features.get("head_pitch", 0.0) or 0.0),
            "head_yaw": float(features.get("head_yaw", 0.0) or 0.0),
            "head_roll": float(features.get("head_roll", 0.0) or 0.0),
        }

        while timestamp_sec >= current_window_end:
            if current_window_rows:
                feats = _create_window_features(current_window_rows)
                window_rows.append(
                    {
                        "window_start_sec": float(current_window_start),
                        "window_end_sec": float(current_window_end),
                        **feats,
                    }
                )
            current_window_rows = []
            current_window_start = current_window_end
            current_window_end += WINDOW_SECONDS

        current_window_rows.append(row)

        if sample_period_s is not None:
            while next_sample_ts <= timestamp_sec + 1e-9:
                next_sample_ts += sample_period_s

    cap.release()

    if current_window_rows:
        feats = _create_window_features(current_window_rows)
        window_rows.append(
            {
                "window_start_sec": float(current_window_start),
                "window_end_sec": float(current_window_end),
                **feats,
            }
        )

    return window_rows, fps


def _predict_windows(window_rows: List[Dict[str, Any]]) -> List[WindowResult]:
    resources = _load_resources()
    feature_columns: List[str] = resources["feature_columns"]
    model: xgb.XGBClassifier = resources["model"]

    if not window_rows:
        return []

    X = np.array([[float(w.get(c, 0.0) or 0.0) for c in feature_columns] for w in window_rows], dtype=np.float32)
    proba = model.predict_proba(X)[:, 1].astype(float)

    results: List[WindowResult] = []
    for w, p in zip(window_rows, proba):
        feats = {k: float(w.get(k, 0.0) or 0.0) for k in feature_columns}
        results.append(
            WindowResult(
                window_start_sec=float(w.get("window_start_sec", 0.0) or 0.0),
                window_end_sec=float(w.get("window_end_sec", 0.0) or 0.0),
                features=feats,
                suspicious_types=_window_suspicious_types(w),
                probability=float(p),
            )
        )
    return results


app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1GB


@app.get("/")
def index() -> Any:
    return render_template_string(
        """
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>Window XGBoost Video Analyzer</title>
          <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
              font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
              background: linear-gradient(135deg, #0f2027 0%, #203a43 45%, #2c5364 100%);
              min-height: 100vh;
              padding: 24px;
              color: #1f2937;
            }
            .container {
              max-width: 1100px;
              margin: 0 auto;
              background: #ffffff;
              border-radius: 18px;
              box-shadow: 0 24px 60px rgba(0, 0, 0, 0.25);
              padding: 36px;
            }
            h1 {
              text-align: center;
              font-size: 2.2rem;
              margin-bottom: 8px;
              color: #0f172a;
            }
            .subtitle {
              text-align: center;
              color: #64748b;
              margin-bottom: 32px;
              font-size: 1.05rem;
            }
            .upload-section {
              border: 2px dashed #2c5364;
              border-radius: 14px;
              padding: 32px;
              text-align: center;
              background: #f8fafc;
              transition: all 0.25s ease;
            }
            .upload-section:hover { background: #f1f5f9; }
            .upload-section.dragover {
              border-color: #0f172a;
              background: #e2e8f0;
            }
            input[type="file"] { display: none; }
            .upload-label {
              display: inline-block;
              padding: 12px 24px;
              background: #0f172a;
              color: #ffffff;
              border-radius: 10px;
              cursor: pointer;
              font-weight: 600;
              transition: transform 0.2s ease;
            }
            .upload-label:hover { transform: scale(1.03); }
            .grid {
              display: grid;
              grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
              gap: 12px;
              margin-top: 20px;
            }
            .field label {
              display: block;
              font-size: 0.9rem;
              color: #475569;
              margin-bottom: 6px;
            }
            .field input {
              width: 100%;
              padding: 10px 12px;
              border: 1px solid #cbd5f5;
              border-radius: 8px;
              font-size: 0.95rem;
            }
            .btn {
              margin-top: 16px;
              background: #2c5364;
              color: #ffffff;
              padding: 12px 22px;
              border: none;
              border-radius: 8px;
              font-size: 1rem;
              cursor: pointer;
              transition: background 0.2s ease;
            }
            .btn:hover { background: #1f3b47; }
            .loading, .error, .result { margin-top: 24px; }
            .loading { display: none; text-align: center; color: #334155; }
            .loading.active { display: block; }
            .spinner {
              border: 4px solid #e2e8f0;
              border-top: 4px solid #0f172a;
              border-radius: 50%;
              width: 46px;
              height: 46px;
              animation: spin 1s linear infinite;
              margin: 0 auto 14px;
            }
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
            .error {
              display: none;
              background: #fee2e2;
              color: #991b1b;
              padding: 14px;
              border-radius: 8px;
              white-space: pre-wrap;
            }
            .error.active { display: block; }
            .result {
              display: none;
              background: #f8fafc;
              border-radius: 12px;
              padding: 20px;
            }
            .result.active { display: block; }
            .report-grid {
              display: grid;
              grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
              gap: 12px;
              margin-top: 12px;
            }
            .card {
              background: #ffffff;
              border: 1px solid #e2e8f0;
              border-radius: 12px;
              padding: 12px 14px;
              color: #0f172a;
            }
            .small {
              color: #64748b;
              font-size: 0.9rem;
              margin-top: 8px;
              text-align: center;
            }
            .mono {
              font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
              font-size: 0.92rem;
              background: #0b1220;
              color: #e2e8f0;
              border-radius: 10px;
              padding: 12px;
              overflow: auto;
              margin-top: 12px;
            }
          </style>
        </head>
        <body>
          <div class="container">
            <h1>Video Fraud Analyzer (1s Windows)</h1>
            <div class="subtitle">Uploads a video, extracts 1-second window features, and returns the maximum suspicious probability.</div>

            <form id="uploadForm" class="upload-section">
              <input id="fileInput" type="file" name="file" accept="video/*" />
              <label class="upload-label" for="fileInput">Choose Video</label>
              <div class="grid">
                <div class="field">
                  <label for="fpsSample">fps_sample (optional)</label>
                  <input id="fpsSample" name="fps_sample" placeholder="e.g. 10" />
                </div>
                <div class="field">
                  <label for="maxSeconds">max_seconds (optional)</label>
                  <input id="maxSeconds" name="max_seconds" placeholder="e.g. 60" />
                </div>
                <div class="field">
                  <label for="prolongedSeconds">prolonged_seconds</label>
                  <input id="prolongedSeconds" name="prolonged_seconds" value="3" />
                </div>
              </div>
              <button class="btn" type="submit">Analyze</button>
              <div class="small">Endpoint: <span class="mono" style="display:inline-block;padding:4px 8px;margin-left:6px;">POST /predict/video</span></div>
            </form>

            <video id="videoPreview" controls style="width: 100%; max-height: 360px; display: none; margin-top: 16px; border-radius: 12px; background: #000;"></video>

            <div id="loading" class="loading">
              <div class="spinner"></div>
              Processing video… this may take a while.
            </div>

            <div id="error" class="error"></div>

            <div id="result" class="result">
              <h3 style="color:#0f172a;">Result</h3>
              <div id="reportGrid" class="report-grid"></div>
              <div id="rawJson" class="mono" style="display:none;"></div>
            </div>
          </div>

          <script>
            const form = document.getElementById("uploadForm");
            const fileInput = document.getElementById("fileInput");
            const loading = document.getElementById("loading");
            const error = document.getElementById("error");
            const result = document.getElementById("result");
            const reportGrid = document.getElementById("reportGrid");
            const rawJson = document.getElementById("rawJson");
            const fpsSample = document.getElementById("fpsSample");
            const maxSeconds = document.getElementById("maxSeconds");
            const prolongedSeconds = document.getElementById("prolongedSeconds");
            const videoPreview = document.getElementById("videoPreview");
            let videoObjectUrl = null;

            function updateVideoPreview() {
              if (!fileInput.files || !fileInput.files.length) {
                videoPreview.style.display = "none";
                if (videoObjectUrl) { URL.revokeObjectURL(videoObjectUrl); videoObjectUrl = null; }
                return;
              }
              const file = fileInput.files[0];
              if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl);
              videoObjectUrl = URL.createObjectURL(file);
              videoPreview.src = videoObjectUrl;
              videoPreview.style.display = "block";
            }

            form.addEventListener("dragover", (e) => { e.preventDefault(); form.classList.add("dragover"); });
            form.addEventListener("dragleave", () => form.classList.remove("dragover"));
            form.addEventListener("drop", (e) => {
              e.preventDefault();
              form.classList.remove("dragover");
              if (e.dataTransfer.files && e.dataTransfer.files.length) fileInput.files = e.dataTransfer.files;
              updateVideoPreview();
            });
            fileInput.addEventListener("change", updateVideoPreview);

            form.addEventListener("submit", (e) => {
              e.preventDefault();
              error.classList.remove("active");
              result.classList.remove("active");
              reportGrid.innerHTML = "";
              rawJson.style.display = "none";

              if (!fileInput.files || !fileInput.files.length) {
                showError("Please choose a video file.");
                return;
              }

              const formData = new FormData();
              formData.append("video", fileInput.files[0]);
              if (fpsSample.value.trim()) formData.append("fps_sample", fpsSample.value.trim());
              if (maxSeconds.value.trim()) formData.append("max_seconds", maxSeconds.value.trim());
              if (prolongedSeconds.value.trim()) formData.append("prolonged_seconds", prolongedSeconds.value.trim());

              loading.classList.add("active");

              fetch("/predict/video", { method: "POST", body: formData })
                .then((response) => response.json())
                .then((data) => {
                  loading.classList.remove("active");
                  if (data.error) {
                    showError(data.error);
                    return;
                  }
                  renderReport(data);
                  rawJson.textContent = JSON.stringify(data, null, 2);
                  rawJson.style.display = "block";
                  result.classList.add("active");
                })
                .catch((err) => {
                  loading.classList.remove("active");
                  showError("Error processing video: " + err.message);
                });
            });

            function showError(message) {
              error.textContent = message;
              error.classList.add("active");
            }

            function renderReport(payload) {
              const exact = payload.exact_problem;
              const exactStr = exact
                ? (exact.type + " — longest " + exact.longest_streak_seconds + "s (total " + exact.total_seconds + "s)")
                : "None";
              const items = [
                ["Windows processed", payload.windows_processed],
                ["Max suspicious probability", payload.max_suspicious_probability],
                ["Decision threshold", payload.decision_threshold],
                ["Exact problem (prolonged)", exactStr],
                ["Suspicious present", (payload.suspicious_present || []).join(", ") || "None"],
                ["Suspicious (above threshold)", (payload.suspicious_present_above_threshold || []).join(", ") || "None"],
                ["Max window start (s)", payload.max_window ? payload.max_window.window_start_sec : "-"],
                ["Max window end (s)", payload.max_window ? payload.max_window.window_end_sec : "-"],
                ["Dominant signal", payload.max_window && payload.max_window.dominant_signal ? (payload.max_window.dominant_signal.type + " (" + payload.max_window.dominant_signal.score + ")") : "-"],
              ];

              items.forEach(([label, value]) => {
                const card = document.createElement("div");
                card.className = "card";
                card.innerHTML = `<strong>${label}:</strong> ${value}`;
                reportGrid.appendChild(card);
              });
            }
          </script>
        </body>
        </html>
        """
    )


@app.get("/health")
def health() -> Any:
    try:
        _load_resources()
        return jsonify(
            {
                "ok": True,
                "endpoints": {
                    "web_interface": "/",
                    "analyze_video": "/analyze-video",
                    "predict_video_compat": "/predict/video",
                },
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/analyze-video")
def analyze_video() -> Any:
    """
    Multipart form-data:
      - file: video file
      - fps_sample (optional): float, target sampling FPS (e.g., 10). If omitted, processes all frames.
      - max_seconds (optional): float, stop processing after N seconds.
      - prolonged_seconds (optional): float, "prolonged" streak length in seconds (default: 3).

    Returns:
      - max suspicious probability across 1s windows
      - suspicious types present in video (rule-based)
      - per-window probabilities + features (optional)
    """
    upload = None
    if "file" in request.files:
        upload = request.files["file"]
    elif "video" in request.files:
        upload = request.files["video"]  # compatibility with app.py-style clients
    if upload is None:
        return jsonify({"error": "Missing form file field: file (or video)"}), 400

    f = upload
    if not f or not f.filename:
        return jsonify({"error": "No file selected"}), 400
    if not _allowed_file(f.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}"}), 400

    fps_sample = request.form.get("fps_sample", None)
    max_seconds = request.form.get("max_seconds", None)
    prolonged_seconds = request.form.get("prolonged_seconds", None)
    try:
        fps_sample_val = float(fps_sample) if fps_sample not in (None, "", "null") else None
    except Exception:
        fps_sample_val = None
    try:
        max_seconds_val = float(max_seconds) if max_seconds not in (None, "", "null") else None
    except Exception:
        max_seconds_val = None
    try:
        prolonged_seconds_val = float(prolonged_seconds) if prolonged_seconds not in (None, "", "null") else 3.0
    except Exception:
        prolonged_seconds_val = 3.0

    filename = secure_filename(f.filename)
    saved_name = f"{uuid.uuid4().hex}_{filename}"
    saved_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_name)

    f.save(saved_path)

    try:
        window_rows, video_fps = _extract_windows_from_video(
            saved_path, fps_sample=fps_sample_val, max_seconds=max_seconds_val
        )
        window_preds = _predict_windows(window_rows)

        resources = _load_resources()
        decision_threshold = float(resources["decision_threshold"])

        if not window_preds:
            return jsonify(
                {
                    "video_fps": video_fps,
                    "windows_processed": 0,
                    "max_suspicious_probability": 0.0,
                    "decision_threshold": decision_threshold,
                    "suspicious_present": [],
                    "exact_problem": None,
                    "problem_summary": [],
                    "max_window": None,
                }
            )

        max_idx = int(np.argmax([w.probability for w in window_preds]))
        max_w = window_preds[max_idx]
        dominant_type, dominant_score = _dominant_suspicious_signal(window_rows[max_idx])
        exact_problem, problem_summary = _compute_problem_summary(
            window_rows,
            ratio_threshold=SUSPICIOUS_RATIO_THRESHOLD,
            prolonged_seconds=prolonged_seconds_val,
        )

        suspicious_any: List[str] = []
        suspicious_by_pred: List[str] = []
        for w in window_preds:
            for t in w.suspicious_types:
                if t not in suspicious_any:
                    suspicious_any.append(t)
            if w.probability >= decision_threshold:
                for t in w.suspicious_types:
                    if t not in suspicious_by_pred:
                        suspicious_by_pred.append(t)

        return jsonify(
            {
                "video_fps": video_fps,
                "windows_processed": len(window_preds),
                "max_suspicious_probability": float(max_w.probability),
                "decision_threshold": decision_threshold,
                "suspicious_present": suspicious_any,
                "suspicious_present_above_threshold": suspicious_by_pred,
                "exact_problem": exact_problem,
                "problem_summary": problem_summary,
                "max_window": {
                    "window_start_sec": float(max_w.window_start_sec),
                    "window_end_sec": float(max_w.window_end_sec),
                    "probability": float(max_w.probability),
                    "suspicious_types": list(max_w.suspicious_types),
                    "dominant_signal": {"type": dominant_type, "score": dominant_score},
                    "features": max_w.features,
                },
                "per_window": [
                    {
                        "window_start_sec": float(w.window_start_sec),
                        "window_end_sec": float(w.window_end_sec),
                        "probability": float(w.probability),
                        "suspicious_types": list(w.suspicious_types),
                    }
                    for w in window_preds
                ],
            }
        )
    finally:
        # Keep uploads directory tidy; delete after processing.
        try:
            if os.path.exists(saved_path):
                os.remove(saved_path)
        except Exception:
            pass


@app.get("/analyze-video")
def analyze_video_get() -> Any:
    return redirect("/")


@app.post("/predict/video")
def predict_video() -> Any:
    return analyze_video()


if __name__ == "__main__":
    # Local dev server
    port = int(os.environ.get("PORT", "5003"))
    app.run(host="0.0.0.0", port=port, debug=False)
