import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (the 'n' stands for nano - it's very fast)
model = YOLO('yolov8n.pt')

# Open the webcam (0 is usually the integrated laptop camera)
cap = cv2.VideoCapture(0)

print("Starting Object Detection... Press 'q' to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Perform detection and tracking
    # persist=True allows the model to remember objects across frames
    results = model.track(frame, persist=True)

    # Visualize the results on the frame (draws boxes and labels)
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow("CodeAlpha - Object Detection & Tracking", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()