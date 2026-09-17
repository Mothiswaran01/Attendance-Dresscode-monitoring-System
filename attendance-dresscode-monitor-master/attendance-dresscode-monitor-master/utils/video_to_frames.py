import cv2
import os

def extract_faces_from_video(video_path, output_folder, detector, frame_skip):

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames:", total_frames)

    frame_count = 0
    saved_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_skip == 0:

            print(f"Processing frame {frame_count}")

            bboxes, _ = detector.detect(frame)

            if bboxes is not None and len(bboxes) > 0:

                box = bboxes[0]

                x1, y1, x2, y2, score = box.astype(int)

                h, w, _ = frame.shape

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                face = frame[y1:y2, x1:x2]

                if face.size == 0:
                    frame_count += 1
                    continue

                face = cv2.resize(face, (112, 112))

                face_name = os.path.join(
                    output_folder,
                    f"face_{saved_count:05d}.jpg"
                )

                cv2.imwrite(face_name, face)

                saved_count += 1

        frame_count += 1

    cap.release()

    print(f"Total faces saved: {saved_count}")