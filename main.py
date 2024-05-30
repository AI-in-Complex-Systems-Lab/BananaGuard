import streamlit as st
from streamlit_webrtc import webrtc_streamer

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
        .slider-label {
            font-size: 18px;
            font-weight: bold;
            color: #333333;
        }
        .stSlider {
            width: 50%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    st.markdown('<div class="title">BananaGuard</div>', unsafe_allow_html=True)

    camera_container = st.empty()

    sensitivity = st.slider('Detection Sensitivity', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    st.markdown('<div class="slider-label">Sensitivity: {}</div>'.format(sensitivity), unsafe_allow_html=True)

   
    with camera_container:
        webrtc_streamer(key="camera")

if __name__ == '__main__':
    main()