from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("yolo11s.pt")  # Replace with your .pt file path

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Perform object detection
    results = model(frame)

    # Display results on the frame
    annotated_frame = results[0].plot()  # Draw bounding boxes and labels on the frame

    # Show the frame
    cv2.imshow("YOLO Object Detection", annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
