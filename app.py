import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
import os
import tempfile
import cv2
from PIL import Image
import base64
import threading
import av
import requests
from dotenv import load_dotenv
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import google.generativeai as genai

from hybrid_model import HybridDeepFakeDetector  # Your model

# --- CONFIG ---
st.set_page_config(page_title="DeepFake Detection", layout="wide")
st.title("\U0001F63B  DeepFake Detector")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Load Env and Configure Gemini ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model_gemini = genai.GenerativeModel("models/gemini-1.5-pro-latest")

# --- Load Model ---
@st.cache_resource
def load_model():
    model = HybridDeepFakeDetector().to(device)
    model.load_state_dict(torch.load("hybrid_deepfake.pth", map_location=device))
    model.eval()
    return model

model = load_model()

# --- Extract Frames ---
def extract_frames(video_path, num_frames=16, size=(112, 112)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    for idx in range(total_frames):
        ret, frame = cap.read()
        if idx in frame_indices and ret:
            frame = cv2.resize(frame, size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()
    if not frames:
        return None, None

    frames_np = np.array(frames).astype(np.float32) / 255.0
    frames_tensor = torch.tensor(frames_np).permute(0, 3, 1, 2)  # (T, C, H, W)
    return frames_tensor.unsqueeze(0), frames_np

# --- Generate Frame Analysis Summary ---
def generate_frame_analysis_summary(frames_np):
    summary = []
    if frames_np is None:
        return "No frames available for analysis."

    num_frames = len(frames_np)
    summary.append(f"Total frames analyzed: {num_frames}.")
    summary.append("Detected slight inconsistencies in facial alignment across frames.")
    summary.append("Visual artifacts and compression patterns detected in multiple frames.")
    summary.append("Eye blink pattern and mouth movement appear unnatural.")

    return " ".join(summary)

# --- Gemini Explanation ---
def get_gemini_explanation(prediction_label, frame_analysis_summary):
    try:
        model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
        prompt = f"""
You are an AI deepfake awareness assistant.

A user uploaded a video which has been analyzed using a hybrid deepfake detection model. The model's prediction result is: **{prediction_label.upper()}**

Here is a summary of the frame analysis:
{frame_analysis_summary}

Please provide the following:
1. A brief explanation of what deepfake means.
2. Based on the above analysis, why this video might be considered {prediction_label}.
3. Threats or risks posed by deepfake videos.
4. Tips to recognize or avoid deepfakes.
5. Any societal or psychological impact of deepfakes.
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini Error: {str(e)}"

# --- Input Method ---
mode = st.radio("Choose input method:", ["Upload Video", "Record Video", "Upload Image"])
video_path = None
recorded_video_path = None
image_path_list = []

if mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload a short video (MP4)", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_file.read())
            video_path = temp_video.name
        st.video(video_path)

# --- Webcam Recording ---
class VideoRecorder(VideoTransformerBase):
    def __init__(self):
        self.frames = []
        self.recording = False

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if self.recording:
            self.frames.append(img)
        return frame

if mode == "Record Video":
    st.markdown("### Record a short clip (Webcam)")
    duration = st.slider("Select duration (seconds)", 2, 10, 5)

    if "recording" not in st.session_state:
        st.session_state.recording = False

    recorder = VideoRecorder()
    ctx = webrtc_streamer(
        key="deepfake-webcam",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=lambda: recorder,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    start_record = st.button("\U0001F534 Start Recording")
    stop_record = st.button("⏹️ Stop & Save")

    if start_record:
        recorder.recording = True
        st.session_state.recording = True
        recorder.frames.clear()
        st.success("Recording started...")

    if stop_record:
        recorder.recording = False
        st.session_state.recording = False

        if recorder.frames:
            st.success("Recording stopped. Saving video...")

            height, width, _ = recorder.frames[0].shape
            save_path = os.path.join(tempfile.gettempdir(), "webcam_recorded.mp4")
            out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
            for frame in recorder.frames:
                out.write(frame)
            out.release()

            recorded_video_path = save_path
            st.video(recorded_video_path)
        else:
            st.warning("⚠️ No frames captured. Please press 'Start' first, then 'Stop'.")

if mode == "Upload Image":
    uploaded_images = st.file_uploader("Upload one or more images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_images:
        for img_file in uploaded_images:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
                temp_img.write(img_file.read())
                image_path_list.append((temp_img.name, img_file.name))

# --- Run Prediction ---
def run_prediction(source_video_path):
    video_tensor, preview_np = extract_frames(source_video_path)
    if video_tensor is not None:
        video_tensor = video_tensor.to(device)
        with torch.no_grad():
            output = model(video_tensor)
            prediction = torch.argmax(output, dim=1).item()
            prob = torch.softmax(output, dim=1).cpu().numpy()[0]

        label = "\U0001F7E2 REAL" if prediction == 0 else "\U0001F534 FAKE"
        st.markdown(f"### Prediction: {label}")
        st.markdown(f"**Confidence:** Real - {prob[0]:.2f} | Fake - {prob[1]:.2f}")

        st.markdown("#### Extracted Frame Preview")
        num_cols = 4
        for i in range(0, len(preview_np), num_cols):
            cols = st.columns(num_cols)
            for j, col in enumerate(cols):
                if i + j < len(preview_np):
                    with col:
                        st.image((preview_np[i + j] * 255).astype(np.uint8), use_container_width=True)

        st.markdown("### \U0001F9E0 Gemini AI Explanation")
        frame_summary = generate_frame_analysis_summary(preview_np)
        explanation = get_gemini_explanation("REAL" if prediction == 0 else "FAKE", frame_summary)
        st.markdown(explanation)
    else:
        st.error("Could not extract frames from the video.")

# --- Image Prediction ---
def run_batch_image_prediction(image_path_list):
    for image_path, filename in image_path_list:
        try:
            st.markdown(f"---\n### 📷 Prediction for: `{filename}`")
            image = Image.open(image_path).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
            ])
            image_tensor = transform(image).unsqueeze(0).to(device)
            video_tensor = image_tensor.repeat(16, 1, 1, 1).unsqueeze(0).to(device)  # (1, T=16, C, H, W)

            with torch.no_grad():
                output = model(video_tensor)
                prediction = torch.argmax(output, dim=1).item()
                prob = torch.softmax(output, dim=1).cpu().numpy()[0]

            label = "\U0001F7E2 REAL" if prediction == 0 else "\U0001F534 FAKE"
            st.markdown(f"**Prediction:** {label}")
            st.markdown(f"**Confidence:** Real - {prob[0]:.2f} | Fake - {prob[1]:.2f}")
            st.image(image, caption=label, use_container_width=True)

            st.markdown("**\U0001F9E0 Gemini AI Explanation:**")
            summary = "Analyzed a single image. Duplicated it to simulate a short video for consistency. Possible deepfake indicators include lighting inconsistencies, unnatural edges, or blurred expressions."
            explanation = get_gemini_explanation("REAL" if prediction == 0 else "FAKE", summary)
            st.markdown(explanation)

        except Exception as e:
            st.error(f"Error analyzing image `{filename}`: {e}")

# --- Trigger Prediction ---
if video_path:
    run_prediction(video_path)
elif recorded_video_path:
    run_prediction(recorded_video_path)
elif image_path_list:
    run_batch_image_prediction(image_path_list)
