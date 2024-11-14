import cv2
import easyocr
import matplotlib.pyplot as plt

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=True)

# Path to the input video (use 0 for webcam)
video_path = 'test_videos\Видео WhatsApp 2024-11-16 в 12.07.51_2c97e2cd.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize VideoWriter for the output video
output_path = 'test_vid'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Threshold for text detection confidence
threshold = 0.1

# Set to store detected license plate texts
detected_license_plates = set()

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect text on the current frame using EasyOCR
    text_results = reader.readtext(frame)

    # Loop through detected text results
    for result in text_results:
        bbox, text, score = result

        if score > threshold and text not in detected_license_plates:
            # Add the license plate text to the set
            detected_license_plates.add(text)

            # Annotate the detected text on the frame
            print(f"Detected license plate: {text} (Confidence: {score:.2f})")
            cv2.rectangle(frame, bbox[0], bbox[2], (0, 255, 0), 5)
            cv2.putText(frame, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

    # Write the annotated frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()

print(f"Annotated video saved to: {output_path}")
