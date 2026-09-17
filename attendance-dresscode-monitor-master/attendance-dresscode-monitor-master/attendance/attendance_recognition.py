import cv2
import numpy as np
import os
import sys
from collections import deque, Counter
import insightface
from insightface.app import FaceAnalysis

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    FACE_RECOGNITION_THRESHOLD,
    TEMPORAL_BUFFER_SIZE,
    TEMPORAL_MIN_VOTES
)
from attendance.face_encoding import load_embeddings, match_face


def load_face_analyzer():
    """Load InsightFace with SCRFD detection + ArcFace recognition."""
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Face analysis initialized.")
    return app


class TemporalBuffer:
    """
    Collects predictions over N frames.
    Confirms identity only when majority vote is reached.
    """
    def __init__(self, size=TEMPORAL_BUFFER_SIZE, min_votes=TEMPORAL_MIN_VOTES):
        self.size      = size
        self.min_votes = min_votes
        self.buffer    = deque(maxlen=size)

    def add(self, name):
        self.buffer.append(name)

    def get_decision(self):
        """
        Returns confirmed name if majority vote reached, else None.
        Ignores 'Unknown' votes.
        """
        if len(self.buffer) < self.size:
            return None  # Not enough frames yet

        valid_votes = [n for n in self.buffer if n != "Unknown"]
        if len(valid_votes) < self.min_votes:
            return None  # Not enough confident votes

        most_common, count = Counter(valid_votes).most_common(1)[0]
        if count >= self.min_votes:
            return most_common
        return None

    def reset(self):
        self.buffer.clear()


def process_frame(frame, face_analyzer, student_embeddings):
    """
    Detect all faces in a frame, match each to a student.
    Returns list of (name, score, bbox) for each detected face.
    """
    results = []

    faces = face_analyzer.get(frame)
    if not faces:
        return results

    for face in faces:
        embedding = face.embedding
        if embedding is None:
            continue

        name, score = match_face(embedding, student_embeddings)

        # Bounding box
        bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]
        results.append((name, score, bbox))

    return results


def draw_results(frame, results):
    """Draw bounding boxes and labels on frame."""
    for name, score, bbox in results:
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} ({score:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


if __name__ == "__main__":
    import glob

    print("Loading face analyzer...")
    analyzer = load_face_analyzer()

    print("Loading embeddings...")
    db = load_embeddings()

    # Find a test image from dataset
    test_images = glob.glob("path/to/face_images/*/*.jpg")
    if not test_images:
        print("No test images found in path/to/face_images/")
        sys.exit(1)

    print(f"\nTesting on {min(5, len(test_images))} sample images...\n")
    print(f"{'Image':<40} {'Matched':<15} {'Score':<10}")
    print("-" * 65)

    for img_path in test_images[:5]:
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        results = process_frame(frame, analyzer, db)
        student_folder = os.path.basename(os.path.dirname(img_path))

        if results:
            for name, score, bbox in results:
                print(f"  {student_folder:<20} → {name:<15} {score:.4f}")
        else:
            print(f"  {student_folder:<20} → No face detected")