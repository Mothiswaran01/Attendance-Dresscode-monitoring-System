from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  

image_path = "path/to/face_image.jpg"  # Example image path

results = model(image_path)

for r in results:
    annotated_frame = r.plot()

    cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Detection", 900, 700)  # adjust size
    cv2.imshow("YOLO Detection", annotated_frame)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
