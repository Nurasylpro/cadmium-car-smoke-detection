import os
from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
import requests
# Initialize Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace().project("car-with-smoke")
model = project.version(2).model

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Connect to your MySQL database
# Image path
image_path = 'test_images/test3.jpeg'

# Load the image
image = cv2.imread(image_path)

# Get predictions from the Roboflow model
result = model.predict(image_path, confidence=20, overlap=3).json()



# Extract bounding boxes
xyxy_boxes = []
for item in result["predictions"]:
    x1 = int(item["x"] - item["width"] / 2)
    y1 = int(item["y"] - item["height"] / 2)
    x2 = int(x1 + item["width"])
    y2 = int(y1 + item["height"])
    xyxy_boxes.append([x1, y1, x2, y2])

# Check if there are any detections
if xyxy_boxes:
    for box in xyxy_boxes:
        x1, y1, x2, y2 = box
        # Crop the detected area
        cropped_region = image[y1:y2, x1:x2]

        # Use EasyOCR to detect text in the cropped region
        text_results = reader.readtext(cropped_region)

        # Set a confidence threshold for text extraction
        threshold = 0.3

        # Iterate through the detected text
        for t_ in text_results:
            bbox, text, score = t_

            if score > threshold:
                # License plate detected
                print(f"Detected License Plate: {text} with confidence {score:.2f}")

                # Annotate the cropped region
                cv2.rectangle(cropped_region, bbox[0], bbox[2], (0, 255, 0), 2)
                cv2.putText(cropped_region, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

                # Insert the license plate into the database


# Define the API endpoint
                url = "http://192.168.149.140:8080/api/add-violation"

                headers = {
                    "Content-Type": "application/json"
                }

                # Define the data to send in the request (as a dictionary)
                data = {
                    "carRegistrationId": text,
                    "location": "testloc"
                }

                # Send a POST request
                response = requests.post(url, json=data, verify=False)  # Using 'json=data' sends the data as JSON

                # Check the response status
                if response.status_code == 200:
                    print("Request was successful!")
                else:
                    print(f"Request failed with status code: {response.status_code}")

        # Display the cropped region
        plt.imshow(cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB))
        plt.title("Detected License Plate from Cropped Area")
        plt.show()
else:
    print("No detections found.")

# Close the database connection

