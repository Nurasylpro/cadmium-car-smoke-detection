import os
from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np
import easyocr
import requests

# Initialize Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace().project("car-with-smoke")
model = project.version(2).model

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Define the API endpoint
url = "http://192.168.149.140:8080/api/add-violation"
headers = {"Content-Type": "application/json"}

# Open the input video
input_video_path = "test_videos/Видео WhatsApp 2024-11-16 в 12.07.51_2c97e2cd.mp4"
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
middle_frame_number = frame_count // 2  # Capture the middle frame for license plate detection

# Initialize the video writer
output_video_path = "output_videos/annotated_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Initialize annotators
bounding_box_annotator = sv.BoxAnnotator()

# Variables to store license plate detection result
license_plate_detected = False
detected_license_text = ""

# Process video frames
current_frame = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Run prediction on the current frame
    result = model.predict(frame, confidence=50, overlap=3).json()

    # Extract bounding boxes
    xyxy_boxes = []
    for item in result["predictions"]:
        x1 = int(item["x"] - item["width"] / 2)
        y1 = int(item["y"] - item["height"] / 2)
        x2 = int(x1 + item["width"])
        y2 = int(y1 + item["height"])
        xyxy_boxes.append([x1, y1, x2, y2])

    # Annotate the frame if detections are found
    annotated_frame = frame.copy()
    if xyxy_boxes:
        for box in xyxy_boxes:
            x1, y1, x2, y2 = box
            # Draw bounding box around detected car with smoke
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Write the annotated frame to the output video
        out.write(annotated_frame)

        # Show the annotated frame
        cv2.imshow("Car with Smoke Detection", annotated_frame)

        # If we reach the middle frame, perform license plate detection
        if current_frame == middle_frame_number and not license_plate_detected:
            for box in xyxy_boxes:
                x1, y1, x2, y2 = box
                cropped_region = frame[y1:y2, x1:x2]

                # Use EasyOCR to detect text in the cropped region
                text_results = reader.readtext(cropped_region, detail=1)

                # Set a confidence threshold for text extraction
                threshold = 0.4

                # Iterate through the detected text
                for t_ in text_results:
                    bbox, text, score = t_
                    if score > threshold:
                        # License plate detected
                        detected_license_text = text
                        print(f"Detected License Plate: {text} with confidence {score:.2f}")
                        license_plate_detected = True

                        # Annotate the cropped region
                        cv2.rectangle(cropped_region, bbox[0], bbox[2], (0, 255, 0), 2)
                        cv2.putText(cropped_region, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

                        # Send the license plate data to the API
                        data = {
                            "carRegistrationId": detected_license_text,
                            "location": "testloc"
                        }

                        # Send a POST request
                        response = requests.post(url, json=data, headers=headers, verify=False)

                        # Check the response status
                        if response.status_code == 200:
                            print("Request was successful!")
                        else:
                            print(f"Request failed with status code: {response.status_code}")

                        # Display the cropped region with annotation
                        cv2.imshow("License Plate Detection", cropped_region)
                        cv2.waitKey(0)  # Pause to show the detected license plate

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    current_frame += 1

# Release video capture and writer, then close windows
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Annotated video saved as '{output_video_path}'")
