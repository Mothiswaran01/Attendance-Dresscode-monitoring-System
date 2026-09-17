import numpy as np
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import FACE_RECOGNITION_THRESHOLD, EMBEDDINGS_PATH

def load_embeddings(path=EMBEDDINGS_PATH):
    """Loads the database from the path defined in config."""
    if not os.path.exists(path):
        print(f"Embedding database not found at {path}")
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)

def match_face(query_embedding, student_embeddings, threshold=FACE_RECOGNITION_THRESHOLD):
    """Match a query embedding against a stored embedding database."""
    best_name  = "Unknown"
    best_score = 0.0

    query = np.array(query_embedding).flatten()
    query = query / (np.linalg.norm(query) + 1e-6)

    for student_name, stored_data in student_embeddings.items():
        if isinstance(stored_data, list):
            for vec in stored_data:
                score = np.dot(query, vec.flatten() / (np.linalg.norm(vec) + 1e-6))
                if score > best_score:
                    best_score = score
                    best_name  = student_name
        else:
            score = np.dot(query, stored_data.flatten() / (np.linalg.norm(stored_data) + 1e-6))
            if score > best_score:
                best_score = score
                best_name  = student_name

    if best_score < threshold:
        return "Unknown", best_score

    return best_name, best_score