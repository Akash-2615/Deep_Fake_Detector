
# 🎭 Deepfake Detection System using Hybrid CNN, ResNet, and BiLSTM

## 🧠 Project Overview

This project is a **Deepfake Detection System** built using a hybrid deep learning model that combines **CNN**, **ResNet**, and **BiLSTM** architectures. It achieves **95%+ accuracy** in detecting real vs fake videos. The system is integrated with **Streamlit** for real-time inference and enhanced with **Gemini API** for AI-powered explanation of the results.

---

## 🚀 Features

- ✅ Real-time Deepfake Detection via Webcam or Uploaded Videos
- 🔀 Hybrid Deep Learning Model (CNN + ResNet + BiLSTM)
- 📈 High Accuracy (>95%) on custom extracted frame datasets
- 🧠 AI-driven Explanation using Gemini API
- 🖼️ Frame-wise visualization of prediction results
- 📸 Frame extraction from videos (preprocessing step)
- 🧪 Evaluation with precision, recall, F1-score

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **PyTorch**
- **OpenCV**
- **Streamlit**
- **Gemini API (Google Generative AI)**
- **NumPy, Matplotlib, Seaborn**
- **Scikit-learn**

---

## 📁 Project Structure

```
deepfake-detector/
├── app.py                    # Streamlit frontend for real-time inference
├── dataset_loader.py        # Dataset loading and preprocessing
├── extractionframes.ipynb   # Notebook for frame extraction from videos
├── hybrid_model.py          # Hybrid model combining CNN, ResNet, BiLSTM
├── simple3dcnn.py           # Custom 3D CNN model
├── resnet_model.py          # ResNet-based image model
├── deepfake_rnn_model.py    # BiLSTM-based sequence model
├── requirements.txt         # List of dependencies
└── README.md                # This file
```

---

## 🧠 Model Architecture

**Hybrid Deepfake Model:**

- 📷 **CNN (3D)**: Captures spatio-temporal video features
- 🧩 **ResNet**: Learns image-level facial features
- 🔁 **BiLSTM**: Captures temporal sequence patterns
- 🧠 Final classifier head for binary classification

All sub-models are trained and combined in `hybrid_model.py`.

---

## 🖼️ Dataset Structure

```
ProcessedFrames/
├── real/
│   ├── video1/
│   │   ├── frame_1.jpg
│   │   └── ...
│   └── ...
├── fake/
│   ├── video2/
│   │   ├── frame_1.jpg
│   │   └── ...
│   └── ...
```

Use `extractionframes.ipynb` to extract frames from videos and structure them.

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/deepfake-detector.git
cd deepfake-detector

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit App

```bash
streamlit run app.py
```

Features in the app:

- Upload or Record a video
- Start/Stop webcam for live detection
- See real-time predictions
- Get Gemini-powered explanations
- Preview extracted frames with results

---

## 🔮 Gemini API Integration

The system integrates Gemini to explain the predictions made by the hybrid model.

To set up:

1. Get API Key from Google Gemini platform.
2. Add your key to environment:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

Or set it directly in the app if needed (not recommended for security).

---

## 📊 Evaluation Metrics

After training, the model was evaluated using:

- **Accuracy**: 95.4%
- **Precision**: 94.7%
- **Recall**: 96.2%
- **F1-Score**: 95.4%

Confusion matrix and ROC-AUC plots can be visualized in the notebook or app.

---

## 📦 Requirements

Here's a basic `requirements.txt` example:

```txt
torch
torchvision
opencv-python
numpy
streamlit
scikit-learn
matplotlib
seaborn
Pillow
google-generativeai
```

---

## 📌 Future Enhancements

- 🎯 Integrate attention mechanism in BiLSTM
- ☁️ Deploy to cloud (AWS/GCP/Heroku)
- 📱 Mobile App (Android/iOS) version
- 🧩 Explainability with Grad-CAM or LIME

---

## 🙌 Credits

Developed by **Akash**  
Machine Learning Engineer | 2025  
Hybrid Deepfake Detection with Real-time Inference and AI-powered Explanation

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change or add.
