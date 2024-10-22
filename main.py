import streamlit as st
import cv2
from ultralytics import YOLO
import time

if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'is_streaming' not in st.session_state:
    st.session_state.is_streaming = False

@st.cache_resource
def load_models():
    gun_model_1 = YOLO('C:/Users/karar/Projects/Website/yolov10.pt')
    gun_model_2 = YOLO('C:/Users/karar/Projects/Website/yolov11.pt')
    yolov10_model = YOLO('yolov10n.pt')
    return {
        "Gun Model 1": gun_model_1,
        "Gun Model 2": gun_model_2,
        "YOLO v10": yolov10_model
    }

gun_models = load_models()

def open_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        st.error(f"Failed to open camera {camera_index}. Please check your camera connection.")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def process_frame(frame, gun_model, yolo_model, sensitivity):

    if sensitivity == 0:
        gun_results = gun_model(frame, conf=0.8)
        yolo_results = []
    else:
        gun_results = gun_model(frame, conf=0.8)
        yolo_results = yolo_model(frame, conf=0.5)

    for result in gun_results + yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label_index = int(box.cls)
            confidence = float(box.conf)

            if result in gun_results:
                label = gun_model.names.get(label_index, "unknown")
            else:
                label = yolo_model.names.get(label_index, "unknown")

            if (sensitivity == 0 and result in gun_results and label == "gun") or \
               (sensitivity == 1 and label in ["gun", "person"]) or \
               (sensitivity == 2):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

def main():
    st.title("BananaGuard - Live Stream Detection")

    sensitivity = st.slider("Sensitivity", 0, 2, 1, key=f"sensitivity_slider")
    selected_gun_model = st.selectbox("Select Gun Model", ["Gun Model 1", "Gun Model 2"], key=f"gun_model_selector")

    start_button = st.button("Start Stream")
    stop_button = st.button("Stop Stream")

    if start_button:
        if not st.session_state.is_streaming:
            if st.session_state.cap is None or not st.session_state.cap.isOpened():
                st.session_state.cap = open_camera(0)

            if st.session_state.cap is not None:
                st.session_state.is_streaming = True

    if stop_button:
        st.session_state.is_streaming = False
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None

    frame_placeholder = st.empty()

    while st.session_state.is_streaming and st.session_state.cap is not None:
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("Failed to capture frame. The camera may have been disconnected.")
            break


        processed_frame = process_frame(frame, gun_models[selected_gun_model], gun_models["YOLO v10"], sensitivity)


        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(frame_rgb, use_column_width=True)

        time.sleep(1/30)

    if not st.session_state.is_streaming and st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None

if __name__ == "__main__":
    main()
