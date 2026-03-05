import cv2
import numpy as np
import xgboost as xgb
import mediapipe as mp
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
import pickle
import matplotlib.pyplot as plt
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

@dataclass
class ExamReport:
    exam_id: str
    peak_score: float
    risk_level: RiskLevel
    peak_frame_id: int
    peak_frame_timestamp: float
    peak_frame_image: np.ndarray
    total_frames: int
    recommendation: str

# ============================================================================
# 3. VIDEO PROCESSING PIPELINE
# ============================================================================

class VideoProcessor:
    """Process video file: extract frames, features, and XGBoost predictions."""
    
    def __init__(self, xgb_model_path: str):
        """
        Initialize with trained XGBoost model.

        Args:
            xgb_model_path: Path to trained XGBoost model (pickle file)
        """
        self.feature_extractor = FeatureExtractor()

        # Load XGBoost model
        with open(xgb_model_path, 'rb') as f:
            self.xgb_model = pickle.load(f)

        print(f"✓ Loaded XGBoost model from {xgb_model_path}")
        
    def process_video(
        self,
        video_path: str,
        exam_id: str = "exam_001",
        fps_sample: int = 1,
        max_frames: int = None
    ) -> Tuple[List[FrameAnalysis], ExamReport]:
        """
        Process entire video and generate fraud detection report.

        Args:
            video_path: Path to exam video file
            exam_id: Identifier for this exam
            fps_sample: Process every Nth frame (1=all, 2=every 2nd, etc.)
            max_frames: Maximum frames to process (None=all)
        
        Returns:
            frame_analyses: List of FrameAnalysis objects
            report: ExamReport with fraud detection results
        """

        print(f"\n{'='*80}")
        print("STARTING VIDEO PROCESSING")
        print(f"{'='*80}")
        print(f"Video: {video_path}")
        print(f"Exam ID: {exam_id}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in video: {total_frames}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Total frames: {total_frames}")
        print(f"FPS: {fps}")
        print(f"Resolution: {width}x{height}")

        frame_analyses = []
        frame_count = 0
        processed_count = 0

        print(f"\n{'='*80}")
        print("PROCESSING FRAMES...")
        print(f"{'='*80}")
        try:
            inference = FraudDetectionInference()
            print("✓ Model loaded successfully\n")            
        except FileNotFoundError as e:
            print(f"✗ Error: {e}")
            print("Please train the model first by running: python src/train.py")
            exit(1)
        
        while True:
            ret, frame = cap.read()
            print("Reading frame:",frame_count)
            if not ret:
                break
            frame_count += 1
            # Sample frames: Basically we are skipping frames based on fps_sample value
            
            if frame_count % fps_sample != 0:
                continue

            if max_frames and processed_count >= max_frames: # Limit frames for testing
                break

            # Extract features
            features = self.feature_extractor.extract_features(frame)
            
            result = inference.predict(features)
            xgb_prob = result['fraud_probability']
            print(f"Prediction Score: {result['fraud_probability']:.4f}")

            # Store analysis

            timestamp = frame_count/fps # store the information

            frame_analyses.append(
                FrameAnalysis(
                    frame_id=frame_count,
                    timestamp=timestamp,
                    exam_id=exam_id,
                    xgboost_probability=float(xgb_prob),
                    frame_image=frame.copy()
                )
            )

            processed_count += 1
            print("Frame ID:",frame_count)
            print("Timestamp:",timestamp)
            print("processed_count:",processed_count)
            # Progress
            if processed_count % 30 == 0: # this is for the last frame
                print(f"Processed {processed_count} frames... " f"(Frame {frame_count}/{total_frames}, "f"Last score: {xgb_prob:.4f})")
        # why it is not reaching to this point:
        # breaking point?
        print("Finished processing all frames.")
        cap.release()
        print(f"\n✓ Processed {processed_count} frames")
        # Generate report
        report = self._generate_report(frame_analyses, exam_id)
        return frame_analyses, report
    
    def _generate_report(
        self,
        frame_analyses: List[FrameAnalysis],
        exam_id: str
    ) -> ExamReport:
        """Generate fraud detection report."""

        probs = [f.xgboost_probability for f in frame_analyses]
        peak_score = max(probs)
        peak_frame_idx = probs.index(peak_score)
        peak_frame = frame_analyses[peak_frame_idx]
        # Classify risk
        if peak_score >= 0.90:
            risk_level = RiskLevel.CRITICAL
            recommendation = (
                "🚨 CRITICAL FRAUD RISK: Exam shows severe suspicious indicators. "
                "RECOMMEND: Immediately investigate and consider invalidating exam."
            )
        elif peak_score >= 0.75:
            risk_level = RiskLevel.HIGH
            recommendation = (
                "⚠️ HIGH FRAUD RISK: Significant suspicious behavior detected. "
                "RECOMMEND: Manual review by proctor, consider re-exam."
            )
        elif peak_score >= 0.50:
            risk_level = RiskLevel.MODERATE
            recommendation = (
                "MODERATE RISK: Some suspicious indicators present. "
                "RECOMMEND: Monitor future exams."
            )
        elif peak_score >= 0.30:
            risk_level = RiskLevel.LOW
            recommendation = (
                "✓ LOW RISK: Minor anomalies detected but within normal range. "
                "RECOMMEND: No action required."
            )
        else:
            risk_level = RiskLevel.MINIMAL
            recommendation = (
                "MINIMAL RISK: Exam appears legitimate. "
                "RECOMMEND: Proceed with confidence."
            )
        
        report = ExamReport(
            exam_id=exam_id,
            peak_score=round(peak_score, 4),
            risk_level=risk_level,
            peak_frame_id=peak_frame.frame_id,
            peak_frame_timestamp=round(peak_frame.timestamp, 2),
            peak_frame_image=peak_frame.frame_image,
            total_frames=len(frame_analyses),
            recommendation=recommendation
        )
        
        return report

# ============================================================================
# 4. REPORT DISPLAY & VISUALIZATION
# ============================================================================

def display_report(report: ExamReport, frame_analyses: List[FrameAnalysis]):
    """Display comprehensive fraud detection report."""
    
    print(f"\n{'='*80}")
    print("EXAM FRAUD DETECTION REPORT")
    print(f"{'='*80}\n")
    
    print(f"Exam ID:                    {report.exam_id}")
    print(f"Peak Suspicious Score:      {report.peak_score:.4f}")
    print(f"Risk Level:                 {report.risk_level.value}")
    print(f"Total Frames Analyzed:      {report.total_frames}")
    print(f"\nMost Suspicious Frame:")
    print(f"  - Frame ID:               {report.peak_frame_id}")
    print(f"  - Timestamp:              {report.peak_frame_timestamp}s")
    print(f"\nRecommendation:")
    print(f"  {report.recommendation}")
    
    print(f"\n{'='*80}")
    print("SCORE DISTRIBUTION")
    print(f"{'='*80}\n")
    
    probs = [f.xgboost_probability for f in frame_analyses]
    print(f"Minimum Score:              {min(probs):.4f}")
    print(f"Maximum Score:              {max(probs):.4f}")
    print(f"Mean Score:                 {np.mean(probs):.4f}")
    print(f"Median Score:               {np.median(probs):.4f}")
    print(f"Std Dev:                    {np.std(probs):.4f}")
    
    # Score histogram
    print(f"\nScore Distribution:")
    ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    for low, high in ranges:
        count = sum(1 for p in probs if low <= p < high)
        pct = count / len(probs) * 100
        bar = "█" * int(pct / 2)
        print(f"  {low:.1f}-{high:.1f}: {bar} {count:4d} ({pct:5.1f}%)")

def visualize_peak_frame(report: ExamReport):
    """Display the peak suspicious frame with annotations."""
    
    frame = report.peak_frame_image.copy()
    
    # Add text annotations
    cv2.putText(
        frame,
        f"Suspicious Score: {report.peak_score:.4f}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),  # Red
        2
    )
    
    cv2.putText(
        frame,
        f"Risk Level: {report.risk_level.value}",
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        2
    )
    
    cv2.putText(
        frame,
        f"Time: {report.peak_frame_timestamp}s",
        (50, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 165, 255),  # Orange
        2
    )
    
    # Add border based on risk level
    color_map = {
        RiskLevel.MINIMAL: (0, 255, 0),    # Green
        RiskLevel.LOW: (0, 255, 255),      # Yellow
        RiskLevel.MODERATE: (0, 165, 255), # Orange
        RiskLevel.HIGH: (0, 127, 255),     # Red-Orange
        RiskLevel.CRITICAL: (0, 0, 255)    # Red
    }
    
    border_color = color_map[report.risk_level]
    border_thickness = 5
    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1] - 1, frame.shape[0] - 1),
        border_color,
        border_thickness
    )
    
    # Display using matplotlib
    plt.figure(figsize=(14, 8))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(
        f"Most Suspicious Frame - Exam {report.exam_id}\n"
        f"Score: {report.peak_score:.4f} | Risk: {report.risk_level.value}",
        fontsize=14,
        fontweight='bold'
    )
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_fraud_scores(frame_analyses: List[FrameAnalysis]):
    """Plot fraud scores over time."""

    timestamps = [f.timestamp for f in frame_analyses]
    probabilities = [f.xgboost_probability for f in frame_analyses]
    
    plt.figure(figsize=(14, 6))
    plt.plot(timestamps, probabilities, linewidth=1.5, color='darkred')
    plt.fill_between(timestamps, probabilities, alpha=0.3, color='red')
    
    # Add threshold lines
    plt.axhline(y=0.5, color='orange', linestyle='--', label='Moderate Threshold (0.5)')
    plt.axhline(y=0.75, color='red', linestyle='--', label='High Threshold (0.75)')
    plt.axhline(y=0.90, color='darkred', linestyle='--', label='Critical Threshold (0.90)')
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Suspicious Probability Score', fontsize=12)
    plt.title('Fraud Detection Score Timeline', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.show()

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main(video_path: str, xgb_model_path: str, exam_id: str = "exam_001"):
    """
    End-to-end fraud detection pipeline.
    
    Args:
        video_path: Path to exam video
        xgb_model_path: Path to trained XGBoost model
        exam_id: Exam identifier
    """
    
    # Initialize processor
    processor = VideoProcessor(xgb_model_path)
    
    # Process video
    frame_analyses, report = processor.process_video(
        video_path=video_path,
        exam_id=exam_id,
        fps_sample=1,  # Process every frame (adjust for speed)
        max_frames=None  # Process all frames (set to 300 for testing)
    )
    
    # Display report
    display_report(report, frame_analyses)

    # Visualize peak frame
    print(f"\nDisplaying peak suspicious frame...")
    visualize_peak_frame(report)
    
    # Plot timeline
    print(f"Plotting fraud score timeline...")
    plot_fraud_scores(frame_analyses)
    return report, frame_analyses

# ============================================================================
# 6. USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    
    report, frame_analyses = main(
        video_path=r"E:\Projects in ML\FRAUD DETECTION SYSTEM FOR THE ONLINE PROCTORED EXAMS\Example Images\input.mp4",  
        xgb_model_path = r"E:\Projects in ML\FRAUD DETECTION SYSTEM FOR THE ONLINE PROCTORED EXAMS\models\xgboost_fraud_detection_model.pkl",
        exam_id="exam_001"
    )
    print(f"\n✓ Fraud detection complete!")
    print(f"Peak Score: {report.peak_score:.4f}")
    print(f"Risk Level: {report.risk_level.value}")