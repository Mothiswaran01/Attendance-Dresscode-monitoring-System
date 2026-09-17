# Face Dataset Preparation

To recognize students, you need a face embedding database — a `.pkl` file mapping student names to their face embeddings.

## Dataset Structure

Organize your dataset as follows:

```
data/
  student_dataset/
    John_Doe/
      frame_00001.jpg
      frame_00002.jpg
      ...
    Jane_Smith/
      frame_00001.jpg
      frame_00002.jpg
      ...
    ...
```

- Each subfolder name becomes the student's display name
- Each folder should contain multiple face crops (at least 5–10 per student)
- Images should be 112×112 face crops ideally

## Option A: Extract Faces from Video

Use `utils/video_to_frames.py` to extract face crops from student videos:

```python
from utils.video_to_frames import extract_faces_from_video
import cv2

detector = cv2.dnn.readNetFromONNX("path/to/scrfd_model.onnx")
extract_faces_from_video("student_video.mp4", "output_folder", detector, frame_skip=5)
```

## Option B: Collect Manually

Place cropped face images directly into each student folder. Ensure:
- Front-facing, well-lit photos
- Multiple angles if possible
- Consistent resolution (112×112 recommended)

## Generate Embeddings

Run `models/face_recognition/generate_database.py`:

```bash
python models/face_recognition/generate_database.py
```

Update the paths in the script before running:

```python
MODEL_PATH = "path/to/scrfd_model.onnx"
DATASET_PATH = "data/student_dataset"
OUTPUT_PATH = "data/student_embeddings.pkl"
```

The script will:
1. Load the SCRFD face detection model
2. Extract ArcFace embeddings for each face image
3. Average embeddings per student
4. L2-normalize and save to a `.pkl` file

## Verify

```bash
python models/face_recognition/test_arcface_embedding.py
```

Update the `model_path` and `image_path` in the script to point to your ArcFace model and a test image.
