import streamlit as st
import cv2
from ultralytics import YOLO
import threading

gun_model = YOLO('C:/Users/karar/Projects/Website/best.pt')
model_high_sensitivity = YOLO('yolov8n.pt')  

model_mid_sensitivity = YOLO('yolov8n.pt')
model_mid_sensitivity.classes = [0] 

gun_class_name = "gun"
person_class_name = "person"

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  

st.title("BananaGuard")
stframe = st.empty()

sensitivity = st.slider("Sensitivity", 0, 2, 2)
sensitivity_labels = {0: "Low", 1: "Medium", 2: "High"}

frame_lock = threading.Lock()
current_frame = None

def capture_frames():
    global current_frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        with frame_lock:
            current_frame = frame

def process_frame(frame, sensitivity):
    gun_results = gun_model(frame)
    
    pre_trained_results = []
    if sensitivity == 2:
        pre_trained_results = model_high_sensitivity(frame)
    elif sensitivity == 1:
        pre_trained_results = model_mid_sensitivity(frame)
    
    combined_results = []
    combined_results.extend(gun_results)
    combined_results.extend(pre_trained_results)
    
    for result in combined_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label_index = box.cls.item() if hasattr(box, 'cls') else -1
            confidence = box.conf.item() if hasattr(box, 'conf') else 0.0 

            label = "unknown"
            if label_index != -1:
                if result in gun_results:
                    label = gun_model.names[label_index] if label_index in gun_model.names else "unknown"
                elif result in pre_trained_results:
                    if sensitivity == 2:
                        label = model_high_sensitivity.names[label_index] if label_index in model_high_sensitivity.names else "unknown"
                    elif sensitivity == 1:
                        label = model_mid_sensitivity.names[label_index] if label_index in model_mid_sensitivity.names else "unknown"
            
            if (sensitivity == 0 and label == gun_class_name) or \
               (sensitivity == 1 and label in [gun_class_name, person_class_name]) or \
               (sensitivity == 2):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()

while True:
    with frame_lock:
        frame = current_frame
    if frame is not None:
        processed_frame = process_frame(frame, sensitivity)
        stframe.image(processed_frame, channels="BGR")

cap.release()
