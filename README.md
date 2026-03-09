# 🔥 IgnisAI — Wildfire Detection App

> AI-powered wildfire detection from aerial and satellite imagery using deep learning.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Keras](https://img.shields.io/badge/Keras-TensorFlow-red?logo=keras)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌎 Overview

**IgnisAI** is a computer vision application that uses a custom-trained convolutional neural network (CNN) to classify images as containing wildfire or not. Upload an aerial or satellite image, and the model returns a real-time prediction — helping support early detection and response efforts.

The project includes:
- A trained Keras model (`myfire_model.keras`)
- A Jupyter notebook documenting the full training pipeline (`myfire_model.ipynb`)
- A web application (`app.py`) for interactive image upload and inference
- Sample images for quick testing

---

## ✨ Features

- 📷 **Image upload** — drag-and-drop or browse to upload any aerial/satellite image
- 🤖 **AI inference** — CNN model predicts wildfire presence with a confidence score
- ⚡ **Fast results** — lightweight model for near-instant predictions
- 🖼️ **Sample images** — included test images to explore the app right away

---

## 🗂️ Project Structure

```
ignisAI-wildfire-app/
├── app.py                  # Web application (Streamlit/Flask)
├── myfire_model.keras      # Pre-trained Keras CNN model
├── myfire_model.ipynb      # Model training notebook
├── requirements.txt        # Python dependencies
└── sample_images/          # Example wildfire & non-wildfire images
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/akhilm1708/ignisAI-wildfire-app.git
   cd ignisAI-wildfire-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   # or, if Flask-based:
   python app.py
   ```

4. **Open in your browser**
   Navigate to `http://localhost:8501` (Streamlit) or `http://localhost:5000` (Flask).

---

## 🧠 Model Details

The wildfire detection model is a CNN trained on labeled aerial imagery. Key details:

| Property | Value |
|---|---|
| Framework | TensorFlow / Keras |
| Input | RGB image |
| Output | Binary classification (fire / no fire) + confidence |
| File | `myfire_model.keras` |

To explore or retrain the model, open the notebook:
```bash
jupyter notebook myfire_model.ipynb
```

---

## 🖼️ Sample Images

The `sample_images/` directory contains example inputs you can use immediately to test the app without needing your own data.

---

## 📦 Dependencies

Key packages (see `requirements.txt` for full list):

- `tensorflow` / `keras` — model loading and inference
- `streamlit` or `flask` — web application framework
- `Pillow` — image processing
- `numpy` — numerical operations

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

Built to support wildfire awareness and early detection efforts. Inspired by the growing need for accessible AI tools in climate and disaster response.
