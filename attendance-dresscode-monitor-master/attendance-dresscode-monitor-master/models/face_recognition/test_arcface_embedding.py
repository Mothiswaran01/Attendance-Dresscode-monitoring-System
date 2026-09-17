import cv2
import os
from embedding_extractor import ArcFaceEmbedder

# Model path
model_path = "path/to/arcface_model.onnx"  # Path to the ArcFace model

# Change this to one of your actual face image paths
image_path = "path/to/face_image.jpg"  # Example image path

if not os.path.exists(image_path):
    print("Image not found. Check the path.")
    exit()

embedder = ArcFaceEmbedder(model_path)

image = cv2.imread(image_path)

if image is None:
    print("Failed to load image.")
    exit()

embedding = embedder.get_embedding(image)

print("Embedding shape:", embedding.shape)
print("First 5 values:", embedding[:5])
