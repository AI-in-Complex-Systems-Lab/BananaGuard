import streamlit as st
import cv2
from ultralytics import YOLO

# Load the custom-trained YOLO model
model = YOLO('C:/Users/karar/Projects/Website/best.pt')

# Class name corresponding to your model
class_name = "gun"

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Streamlit app
st.title("YOLO Object Detection")
stframe = st.empty()

def get_frame():
    ret, frame = cap.read()
    if not ret:
        st.error("Error: Could not read frame.")
        return None
    return frame

def process_frame(frame):
    results = model(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = class_name  # Since there's only one class, label is always "gun"

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display the label
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

while True:
    frame = get_frame()
    if frame is not None:
        processed_frame = process_frame(frame)
        stframe.image(processed_frame, channels="BGR")

cap.release()
