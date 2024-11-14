import os
import cv2
import easyocr
import numpy as np
from roboflow import Roboflow
import requests

# Initialize Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace().project("car-with-smoke")
model = project.version(2).model

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Video input path (use 0 for webcam)
video_path = 'test_videos/test1.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize VideoWriter for output video
output_path = 'annotated_output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Skipping frames for faster processing
skip_frames = 5
frame_count = 0
text_confidence_threshold = 0.4
detected_license_plates = set()

# API endpoint for license plate violation
api_url = "http://192.168.149.164:8080/api/add-violation"
headers = {"Content-Type": "application/json"}

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames for faster processing
    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    # Save the current frame temporarily
    temp_frame_path = "temp_frame.jpg"
    cv2.imwrite(temp_frame_path, frame)

    # Get predictions from the Roboflow model
    result = model.predict(temp_frame_path, confidence=50, overlap=3).json()

    # Extract bounding boxes
    xyxy_boxes = []
    for item in result["predictions"]:
        x1 = int(item["x"] - item["width"] / 2)
        y1 = int(item["y"] - item["height"] / 2)
        x2 = int(x1 + item["width"])
        y2 = int(y1 + item["height"])
        xyxy_boxes.append([x1, y1, x2, y2])

    # Process each detected car
    for box in xyxy_boxes:
        x1, y1, x2, y2 = box

        # Annotate the detected car on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Take the middle frame of the detection for license plate recognition
        cropped_region = frame[y1:y2, x1:x2]

        # Detect text in the cropped region using EasyOCR
        text_results = reader.readtext(cropped_region)

        # Check detected text
        for t_ in text_results:
            bbox, text, score = t_

            if score > text_confidence_threshold and text not in detected_license_plates:
                # License plate detected
                detected_license_plates.add(text)
                print(f"Detected License Plate: {text} with confidence {score:.2f}")

                # Annotate the license plate text
                cv2.rectangle(cropped_region, bbox[0], bbox[2], (0, 255, 0), 2)
                cv2.putText(cropped_region, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

                # Send detected license plate to the API
                data = {
                    "carRegistrationId": text,
                    "location": "testloc"
                }
                response = requests.post(api_url, json=data, headers=headers, verify=False)
                if response.status_code == 200:
                    print("Request was successful!")
                else:
                    print(f"Request failed with status code: {response.status_code}")

    # Write the annotated frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()

print(f"Annotated video saved to: {output_path}")
