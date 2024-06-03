import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import torch
from model_code import ObjectDetectionModel
import time

class VideoTransformer(VideoTransformerBase):
    def __init__(self, model):
        self.model = model

    def transform(self, frame):
        start_time = time.time()

        img = frame.to_ndarray(format="bgr24")

        # Preprocess the frame
        processed_frame = self.model.preprocess_frame(img)

        # Perform detection
        detection_results = self.model.predict(processed_frame)

        # Visualize detections
        annotated_frame = self.model.visualize_detections(img, detection_results)

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return annotated_frame

def main():
    st.set_page_config(page_title='BananaGuard', page_icon=':camera:')
    st.markdown(
        """
        <style>
        .reportview-container {
            background-color: #f0f0f0;
        }
        .title {
            font-size: 38px;
            font-weight: bold;
            color: #333333;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="title">BananaGuard</div>', unsafe_allow_html=True)

    # Load your trained model
    model_path = "C:\\Users\\karar\\Projects\\Website\\trained_model.pt"
    model = ObjectDetectionModel(model_path)

    webrtc_streamer(
        key="camera",
        video_transformer_factory=lambda: VideoTransformer(model),
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 15, "max": 15},
            }
        }
    )

if __name__ == '__main__':
    main()
