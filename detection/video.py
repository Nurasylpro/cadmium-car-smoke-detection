import os
from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np

# Initialize Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace().project("car-with-smoke")
model = project.version(2).model

# Open input video
input_video_path = "input_video.mp4"
output_video_path = "output_video_with_annotations.mp4"

cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Set up annotators
label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Run prediction on the current frame
    cv2.imwrite("temp_frame.jpg", frame)  # Save current frame as temp image
    result = model.predict("temp_frame.jpg", confidence=40, overlap=30).json()

    # Extract bounding boxes, confidences, and labels
    xyxy_boxes = []
    confidences = []
    labels = []
    for item in result["predictions"]:
        x1 = item["x"] - item["width"] / 2
        y1 = item["y"] - item["height"] / 2
        x2 = x1 + item["width"]
        y2 = y1 + item["height"]
        xyxy_boxes.append([x1, y1, x2, y2])  # Convert to [x1, y1, x2, y2] format
        confidences.append(item["confidence"])
        labels.append(item["class"])

    # Check if there are any detections
    if len(xyxy_boxes) > 0:
        # Convert lists to numpy arrays
        xyxy_boxes = np.array(xyxy_boxes)
        confidences = np.array(confidences)

        # Map each unique label to an integer ID
        unique_labels = list(set(labels))
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        class_ids = np.array([label_to_id[label] for label in labels])

        # Create detections object with integer class_ids
        detections = sv.Detections(xyxy=xyxy_boxes, confidence=confidences, class_id=class_ids)

        # Annotate frame with bounding boxes and labels
        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # Add title text at the top of the frame
        if "car with smoke" in labels:
            title_text = "Car with Smoke Detected"
        else:
            title_text = "No Smoke Detected"
    else:
        # If no detections, use original frame and set "No Smoke Detected" text
        annotated_frame = frame
        title_text = "No Smoke Detected"

    cv2.putText(annotated_frame, title_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # Write annotated frame to output video
    out.write(annotated_frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
