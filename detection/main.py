import os
from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np

# Initialize Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace().project("car-with-smoke")
model = project.version(2).model

# Get predictions from the model
result = model.predict("test_images/test4.jpg", confidence=50, overlap=30).json()

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

# Ensure there are detections before proceeding
if xyxy_boxes:
    # Convert lists to numpy arrays
    xyxy_boxes = np.array(xyxy_boxes)
    confidences = np.array(confidences)
    
    # Map each unique label to an integer ID
    unique_labels = list(set(labels))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    class_ids = np.array([label_to_id[label] for label in labels])
    
    # Create detections object with integer class_ids
    detections = sv.Detections(xyxy=xyxy_boxes, confidence=confidences, class_id=class_ids)
    
    # Annotate image
    label_annotator = sv.LabelAnnotator()
    bounding_box_annotator = sv.BoxAnnotator()
    image = cv2.imread("test_images/test4.jpg")
    
    # Apply annotations
    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    
    # Plot image
    sv.plot_image(image=annotated_image, size=(16, 16))
else:
    print("No detections found.")
