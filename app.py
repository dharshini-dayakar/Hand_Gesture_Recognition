import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import mediapipe as mp
import pickle

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils


class GestureRecognizer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                data = []

                for lm in hand_landmarks.landmark:
                    data.append(lm.x)
                    data.append(lm.y)

                prediction = model.predict([data])[0]

                h, w, c = img.shape

                x = int(hand_landmarks.landmark[0].x * w)
                y = int(hand_landmarks.landmark[0].y * h)

                mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                cv2.putText(
                    img,
                    f"{prediction}",
                    (x, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

        return img


st.set_page_config(page_title="Hand Gesture Recognition", layout="centered")

st.title("✋ Hand Gesture Recognition")
st.write("Show one of the trained gestures:")
st.write("👍 Thumbs Up | 👎 Thumbs Down | ✊ Fist | ✌ Peace | ✋ Palm")

webrtc_streamer(
    key="gesture-recognition",
    video_transformer_factory=GestureRecognizer,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)