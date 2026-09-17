import os
import cv2
import numpy as np
import pickle
from embedding_extractor import FaceRecognitionPipeline


# Replace these placeholders with your own local paths before running.
MODEL_PATH = "path/to/scrfd_model.onnx"
DATASET_PATH = "path/to/student_dataset"
OUTPUT_PATH = "path/to/student_embeddings.pkl"

print("\nLoading face recognition model...\n")
pipeline = FaceRecognitionPipeline(MODEL_PATH)

student_database = {}

print("\nGenerating aligned embeddings...\n")

for student_name in os.listdir(DATASET_PATH):
    student_folder = os.path.join(DATASET_PATH, student_name)

    if not os.path.isdir(student_folder):
        continue

    embeddings = []

    for image_name in os.listdir(student_folder):
        image_path = os.path.join(student_folder, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Skipped unreadable image: {image_path}")
            continue

        embedding = pipeline.get_embedding(image)
        
        if embedding is not None:
            embeddings.append(embedding)

    if len(embeddings) > 0:
        avg_embedding = np.mean(embeddings, axis=0)
        
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        student_database[student_name] = avg_embedding
        print(f"{student_name}: {len(embeddings)} valid frames processed & averaged.")
    else:
        print(f"{student_name}: No faces detected in any frames!")


print("\nSaving database...")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(student_database, f)

print(f"Database saved successfully to: {OUTPUT_PATH}")
print(f"Total students registered: {len(student_database)}")