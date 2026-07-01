import base64
import os
import pickle
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import MODEL_PATH
from src.feature_extraction import FeatureExtractor
from src.inference import FraudDetectionInference

# ============================================================================
# 1. DATA STRUCTURES
# ============================================================================

class RiskLevel(Enum):
    MINIMAL = "MINIMAL"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class FrameAnalysis:
    frame_id: int
    timestamp: float
    exam_id: str
    xgboost_probability: float
    frame_image: np.ndarray = None
    suspicious_events: List[str] = field(default_factory=list)
    suspicious_meta: Dict = field(default_factory=dict)

@dataclass
class ExamReport:
    exam_id: str
    peak_score: float
    risk_level: RiskLevel
    peak_frame_id: int
    peak_frame_timestamp: float
    peak_frame_image: np.ndarray
    peak_frame_events: List[str]
    total_frames: int
    recommendation: str
    
# ============================================================================
# 2. VIDEO PROCESSING PIPELINE
# ============================================================================

def _collect_suspicious_events(
    features: Dict, *, phone_bbox: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[List[str], Dict]:
    events: List[str] = []
    meta: Dict = {}

    if int(features.get("phone_present", 0) or 0) == 1:
        events.append("Mobile phone detected")
        if phone_bbox is not None:
            meta["phone_bbox"] = phone_bbox

    if int(features.get("hand_obj_interaction", 0) or 0) == 1:
        events.append("Hand-object interaction")

    face_present = int(features.get("face_present", 0) or 0)
    if face_present == 0:
        events.append("Face not visible")

    try:
        if float(features.get("no_of_face", 0) or 0) > 1:
            events.append("Multiple faces detected")
    except Exception:
        pass

    head_pose = (features.get("head_pose") or "None").lower()
    if head_pose in {"left", "right"}:
        events.append(f"Looking {head_pose}")
    elif head_pose == "down":
        events.append("Looking down")

    gaze_direction = (features.get("gaze_direction") or "None").lower()
    if gaze_direction.endswith("left"):
        events.append("Gaze left")
    elif gaze_direction.endswith("right"):
        events.append("Gaze right")

    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for e in events:
        if e not in seen:
            deduped.append(e)
            seen.add(e)

    return deduped, meta


def _annotate_peak_frame(
    frame: np.ndarray, suspicious_events: List[str], suspicious_meta: Dict
) -> np.ndarray:
    if frame is None:
        return frame
    if not suspicious_events and not suspicious_meta:
        return frame

    annotated = frame.copy()
    height, width = annotated.shape[:2]

    phone_bbox = suspicious_meta.get("phone_bbox")
    if phone_bbox and len(phone_bbox) == 4:
        x1, y1, x2, y2 = [int(v) for v in phone_bbox]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 > x1 and y2 > y1:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                annotated,
                "MOBILE",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

    # Draw a readable top-left panel with the detected suspicious events.
    panel_lines = ["Peak Frame - Detected:"] + [f"- {e}" for e in suspicious_events]
    if len(panel_lines) == 1:
        return annotated

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    line_height = 22
    padding = 10

    max_line_w = 0
    for line in panel_lines:
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_line_w = max(max_line_w, tw)

    panel_w = min(width - 20, max_line_w + padding * 2)
    panel_h = min(height - 20, padding * 2 + line_height * len(panel_lines))
    x0, y0 = 10, 10
    x1, y1 = x0 + panel_w, y0 + panel_h

    overlay = annotated.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
    annotated = cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0)
    cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 0, 255), 2)

    y = y0 + padding + 16
    for idx, line in enumerate(panel_lines):
        color = (255, 255, 255) if idx == 0 else (0, 200, 255)
        cv2.putText(annotated, line, (x0 + padding, y), font, font_scale, color, thickness)
        y += line_height

    return annotated


class VideoProcessor:
    """Process video file: extract frames, features, and fraud predictions."""

    def __init__(self, xgb_model_path: Optional[str] = None):
        self.feature_extractor = FeatureExtractor()
        # Load XGBoost model (kept for consistency with training artifacts)
        model_path = xgb_model_path or MODEL_PATH
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.xgb_model = pickle.load(f)
            print(f"Loaded XGBoost model from {model_path}")
        else:
            self.xgb_model = None
            print(f"Warning: XGBoost model not found at {model_path}")

    def process_video(
        self,
        video_path: str,
        exam_id: str = "exam_001",
        fps_sample: float = 20.0,
        max_frames: Optional[int] = None,
    ) -> Tuple[List[FrameAnalysis], ExamReport]:
        """Process a video and generate a fraud detection report."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
        frame_analyses: List[FrameAnalysis] = []
        frame_count = 0
        processed_count = 0
        inference = FraudDetectionInference()

        # `fps_sample` is treated as the target sampling rate (frames per second).
        target_fps = float(fps_sample) if fps_sample else 20.0
        if target_fps <= 0:
            target_fps = 20.0
        sample_period_s = 1.0 / target_fps
        next_sample_ts = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            timestamp = frame_count / fps
            if timestamp + 1e-9 < next_sample_ts:
                continue
            if max_frames and processed_count >= max_frames:
                break

            features = self.feature_extractor.extract_features(frame)
            phone_bbox = None
            if int(features.get("phone_present", 0) or 0) == 1:
                pf = getattr(self.feature_extractor, "phone_features", {}) or {}
                if int(pf.get("phone_present", 0) or 0) == 1:
                    phone_bbox = (
                        int(pf.get("phone_x1", 0)),
                        int(pf.get("phone_y1", 0)),
                        int(pf.get("phone_x2", 0)),
                        int(pf.get("phone_y2", 0)),
                    )

            suspicious_events, suspicious_meta = _collect_suspicious_events(
                features, phone_bbox=phone_bbox
            )
            result = inference.predict(features)
            xgb_prob = result["fraud_probability"]

            frame_analyses.append(
                FrameAnalysis(
                    frame_id=frame_count,
                    timestamp=timestamp,
                    exam_id=exam_id,
                    xgboost_probability=float(xgb_prob),
                    frame_image=frame.copy(),
                    suspicious_events=suspicious_events,
                    suspicious_meta=suspicious_meta,
                )
            )
            processed_count += 1

            # Move to the next sample timestamp (handle multiple periods per frame if needed).
            while next_sample_ts <= timestamp + 1e-9:
                next_sample_ts += sample_period_s

        cap.release()
        report = self._generate_report(frame_analyses, exam_id)
        return frame_analyses, report

    def _generate_report(
        self, frame_analyses: List[FrameAnalysis], exam_id: str
    ) -> ExamReport:
        probs = [f.xgboost_probability for f in frame_analyses]
        if probs:
            peak_score = max(probs)
            peak_frame_idx = probs.index(peak_score)
            peak_frame = frame_analyses[peak_frame_idx]
        else:
            peak_score = 0.0
            peak_frame = FrameAnalysis(
                frame_id=0,
                timestamp=0.0,
                exam_id=exam_id,
                xgboost_probability=0.0,
                frame_image=None,
            )

        if peak_score >= 0.90:
            risk_level = RiskLevel.CRITICAL
            recommendation = (
                "CRITICAL FRAUD RISK: Severe suspicious indicators detected. "
                "Recommend immediate investigation."
            )
        elif peak_score >= 0.75:
            risk_level = RiskLevel.HIGH
            recommendation = (
                "HIGH FRAUD RISK: Significant suspicious behavior detected. "
                "Recommend manual review by proctor."
            )
        elif peak_score >= 0.50:
            risk_level = RiskLevel.MODERATE
            recommendation = (
                "MODERATE RISK: Some suspicious indicators present. "
                "Recommend monitoring."
            )
        elif peak_score >= 0.30:
            risk_level = RiskLevel.LOW
            recommendation = (
                "LOW RISK: Minor anomalies detected but within normal range."
            )
        else:
            risk_level = RiskLevel.MINIMAL
            recommendation = "MINIMAL RISK: Exam appears legitimate."

        return ExamReport(
            exam_id=exam_id,
            peak_score=round(peak_score, 4),
            risk_level=risk_level,
            peak_frame_id=peak_frame.frame_id,
            peak_frame_timestamp=round(peak_frame.timestamp, 2),
            peak_frame_image=_annotate_peak_frame(
                peak_frame.frame_image, peak_frame.suspicious_events, peak_frame.suspicious_meta
            )
            if peak_frame.frame_image is not None
            else None,
            peak_frame_events=peak_frame.suspicious_events,
            total_frames=len(frame_analyses),
            recommendation=recommendation,
        )
    

def _build_report_payload(report: ExamReport, frame_analyses: List[FrameAnalysis]) -> dict:
    probs = [f.xgboost_probability for f in frame_analyses]
    stats = {
        "min": float(min(probs)) if probs else 0.0,
        "max": float(max(probs)) if probs else 0.0,
        "mean": float(np.mean(probs)) if probs else 0.0,
        "median": float(np.median(probs)) if probs else 0.0,
        "std": float(np.std(probs)) if probs else 0.0,
    }
    return {
        "exam_id": report.exam_id,
        "peak_score": report.peak_score,
        "risk_level": report.risk_level.value,
        "peak_frame_id": report.peak_frame_id,
        "peak_frame_timestamp": report.peak_frame_timestamp,
        "peak_frame_events": report.peak_frame_events,
        "total_frames": report.total_frames,
        "recommendation": report.recommendation,
        "score_stats": stats,
    }


def _encode_image_b64(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        return ""
    return base64.b64encode(buffer).decode("utf-8")


# ============================================================================
# 3. FLASK APP
# ============================================================================


app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"mp4", "avi", "mov", "mkv"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

processor: Optional[VideoProcessor] = None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config[
        "ALLOWED_EXTENSIONS"
    ]


def init_processor() -> bool:
    global processor
    try:
        processor = VideoProcessor()
        return True
    except Exception as exc:
        print(f"Error initializing processor: {exc}")
        processor = None
        return False


init_processor()


@app.route("/")
def index():
    
    return render_template_string(
        """
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>Fraud Detection Video Processing</title>
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
              border-radius: 10px;
              padding: 12px;
              font-size: 0.95rem;
            }
            .card strong { color: #0f172a; }
            .peak-frame {
              margin-top: 16px;
              text-align: center;
            }
            .peak-frame img {
              max-width: 100%;
              border-radius: 10px;
              box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
            }
          </style>
        </head>
        <body>
          <div class="container">
            <h1>AI-BASED FRAUD DETECTION SYSTEM FOR PROCTORING ONLINE EXAMS</h1>
            <p class="subtitle">Upload a proctored exam video and receive a session-level suspiciousness report.</p>

            <div class="upload-section" id="uploadSection">
              <h2>Upload Exam Video</h2>
              <p style="margin: 12px 0; color: #64748b;">Drop a video or use the button below</p>
              <label for="videoInput" class="upload-label">Choose Video</label>
              <input type="file" id="videoInput" accept="video/*">

              <div class="grid">
                <div class="field">
                  <label for="examId">Exam ID</label>
                  <input type="text" id="examId" placeholder="exam_001" value="exam_001">
                </div>
                <div class="field">
                  <label for="fpsSample">Sample Rate (frames/sec)</label>
                  <input type="number" id="fpsSample" value="20" min="1">
                </div>
                <div class="field">
                  <label for="maxFrames">Max Frames (optional)</label>
                  <input type="number" id="maxFrames" placeholder="e.g., 300">
                </div>
                <div class="field">
                  <label for="includePeak">Include Peak Frame</label>
                  <input type="checkbox" id="includePeak" checked>
                </div>
              </div>

              <button class="btn" id="uploadBtn">Analyze Video</button>
              <p style="margin-top: 10px; color: #94a3b8; font-size: 0.9rem;">
                Supported: MP4, AVI, MOV, MKV
              </p>
            </div>

            <div class="loading" id="loading">
              <div class="spinner"></div>
              <p>Processing video and generating report...</p>
            </div>

            <div class="error" id="error"></div>

            <div class="result" id="result">
              <h3>Suspiciousness Report</h3>
              <div class="report-grid" id="reportGrid"></div>
              <div class="peak-frame" id="peakFrame"></div>
            </div>
          </div>

          <script>
            const videoInput = document.getElementById("videoInput");
            const uploadSection = document.getElementById("uploadSection");
            const uploadBtn = document.getElementById("uploadBtn");
            const loading = document.getElementById("loading");
            const error = document.getElementById("error");
            const result = document.getElementById("result");
            const reportGrid = document.getElementById("reportGrid");
            const peakFrame = document.getElementById("peakFrame");

            uploadSection.addEventListener("dragover", (e) => {
              e.preventDefault();
              uploadSection.classList.add("dragover");
            });

            uploadSection.addEventListener("dragleave", () => {
              uploadSection.classList.remove("dragover");
            });

            uploadSection.addEventListener("drop", (e) => {
              e.preventDefault();
              uploadSection.classList.remove("dragover");
              const files = e.dataTransfer.files;
              if (files.length > 0) {
                videoInput.files = files;
              }
            });

            uploadBtn.addEventListener("click", () => {
              const file = videoInput.files[0];
              if (!file) {
                showError("Please select a video file.");
                return;
              }

              const formData = new FormData();
              formData.append("video", file);
              formData.append("exam_id", document.getElementById("examId").value || "exam_001");
              formData.append("fps_sample", document.getElementById("fpsSample").value || "1");
              formData.append("max_frames", document.getElementById("maxFrames").value || "");
              formData.append("include_peak_frame", document.getElementById("includePeak").checked ? "1" : "0");

              loading.classList.add("active");
              result.classList.remove("active");
              error.classList.remove("active");

              fetch("/predict/video", { method: "POST", body: formData })
                .then((response) => response.json())
                .then((data) => {
                  loading.classList.remove("active");
                  if (data.error) {
                    showError(data.error);
                    return;
                  }
                  renderReport(data.report);
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

            function renderReport(report) {
              reportGrid.innerHTML = "";
              peakFrame.innerHTML = "";

              const items = [
                ["Exam ID", report.exam_id],
                ["Risk Level", report.risk_level],
                ["Peak Score", report.peak_score],
                ["Peak Frame", report.peak_frame_id],
                ["Peak Timestamp (s)", report.peak_frame_timestamp],
                ["Total Frames", report.total_frames],
                ["Recommendation", report.recommendation],
                ["Score Min", report.score_stats.min],
                ["Score Max", report.score_stats.max],
                ["Score Mean", report.score_stats.mean],
                ["Score Median", report.score_stats.median],
                ["Score Std", report.score_stats.std]
              ];

              items.forEach(([label, value]) => {
                const card = document.createElement("div");
                card.className = "card";
                card.innerHTML = `<strong>${label}:</strong> ${value}`;
                reportGrid.appendChild(card);
              });

              if (report.peak_frame_image) {
                const events = Array.isArray(report.peak_frame_events) ? report.peak_frame_events : [];
                const eventsHtml = events.length
                  ? `<div style="margin: 10px 0; color: #334155; font-size: 0.95rem;"><strong>Detected:</strong> ${events.join(", ")}</div>`
                  : "";
                peakFrame.innerHTML = `
                  <h4>Peak Suspicious Frame</h4>
                  ${eventsHtml}
                  <img src="data:image/jpeg;base64,${report.peak_frame_image}" alt="Peak Frame">
                `;
              }
            }
          </script>
        </body>
        </html>
        """
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy" if processor is not None else "processor_not_loaded",
            "processor_loaded": processor is not None,
            "endpoints": {"web_interface": "/", "video_prediction": "/predict/video"},
        }
    )


@app.route("/predict/video", methods=["POST"])
def predict_video():
    if processor is None:
        return jsonify({"error": "Processor not loaded. Check model files."}), 503
    
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Allowed: MP4, AVI, MOV, MKV"}), 400

    exam_id = request.form.get("exam_id", "exam_001")
    fps_sample = float(request.form.get("fps_sample", 20))
    max_frames_raw = request.form.get("max_frames", "").strip()
    max_frames = int(max_frames_raw) if max_frames_raw else None
    include_peak_frame = request.form.get("include_peak_frame", "0") == "1"

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    try:
        frame_analyses, report = processor.process_video(
            video_path=save_path,
            exam_id=exam_id,
            fps_sample=fps_sample,
            max_frames=max_frames,
        )
        payload = _build_report_payload(report, frame_analyses)
        if include_peak_frame and report.peak_frame_image is not None:
            payload["peak_frame_image"] = _encode_image_b64(report.peak_frame_image)
        return jsonify({"success": True, "report": payload})
    except Exception as exc:
        return jsonify({"error": f"Video processing error: {exc}"}), 500

if __name__ == "__main__":
    print("=" * 60)
    print("Fraud Detection Video API")
    print("=" * 60)
    print("Access the web interface at: http://localhost:5000")
    print("Health check: http://localhost:5000/health")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
