import os
from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np

# Инициализация Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace().project("car-with-smoke")
model = project.version(2).model

# Открываем входное видео
input_video_path = "test_videos\Видео WhatsApp 2024-11-15 в 22.56.26_22332a78.mp4"
output_video_path = "output_video_with_annotations.mp4"

cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Определение видеозаписи
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Настройка аннотаторов
label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Конец видео

    # Запуск предсказания прямо на кадре
    result = model.predict(frame, confidence=40, overlap=30).json()

    # Извлечение границ, вероятностей и меток
    xyxy_boxes = []
    confidences = []
    labels = []
    for item in result["predictions"]:
        x1 = item["x"] - item["width"] / 2
        y1 = item["y"] - item["height"] / 2
        x2 = x1 + item["width"]
        y2 = y1 + item["height"]
        xyxy_boxes.append([x1, y1, x2, y2])
        confidences.append(item["confidence"])
        labels.append(item["class"])

    # Проверка на наличие обнаружений
    if len(xyxy_boxes) > 0:
        xyxy_boxes = np.array(xyxy_boxes)
        confidences = np.array(confidences)
        unique_labels = list(set(labels))
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        class_ids = np.array([label_to_id[label] for label in labels])

        detections = sv.Detections(xyxy=xyxy_boxes, confidence=confidences, class_id=class_ids)

        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)




    out.write(annotated_frame)

# Освобождение ресурсов
cap.release()
out.release()
