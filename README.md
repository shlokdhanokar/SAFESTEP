<p align="center">
  <h1 align="center">🦯 SafeStep</h1>
  <p align="center">
    <strong>Real-Time Obstacle Detection for Visually Impaired Individuals</strong>
  </p>
  <p align="center">
    <a href="https://github.com/shlokdhanokar/SAFESTEP/actions"><img src="https://github.com/shlokdhanokar/SAFESTEP/actions/workflows/python-app.yml/badge.svg" alt="Build Status"></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.7+-blue.svg" alt="Python 3.7+"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  </p>
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#tech-stack">Tech Stack</a> •
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a> •
    <a href="#how-it-works">How It Works</a> •
    <a href="#future-enhancements">Roadmap</a>
  </p>
</p>

---



## 📖 About

There are nearly **285 million** visually impaired people worldwide, many of whom face the daily challenge of detecting obstacles in their path — leading to potential injuries and reduced independence.

**SafeStep** is a wearable assistive device that uses computer vision to detect obstacles in real-time and alerts the user through audio notifications. It can be attached to everyday accessories like **glasses, hats, or walking sticks**, giving users the confidence to navigate their surroundings safely.

---

## ✨ Features

- 🎥 **Real-Time Detection** — Continuous video feed processing for instant obstacle awareness
- 🔊 **Audio Alerts** — Text-to-speech notifications when an object is detected
- 📦 **Bounding Box Visualization** — Visual overlay highlighting detected objects on-screen
- 🪶 **Lightweight** — Runs on minimal hardware with low computational overhead
- 🔌 **Portable** — Designed to attach to glasses, hats, caps, or walking sticks

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python** | Core application language |
| **OpenCV** (`cv2`) | Video capture, color-space conversion, contour detection |
| **NumPy** | Numerical array operations for HSV thresholds |
| **pyttsx3** | Offline text-to-speech engine for audio alerts |

### Hardware Requirements

- Webcam or USB camera module
- Speaker or earphone (for audio alerts)
- Computer or Raspberry Pi to run the script

---

## 📦 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- A connected webcam/camera

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/shlokdhanokar/safestep.git
   cd safestep
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

4. **Quit** — Press `Q` to stop the detection feed.

---

## 🚀 Usage

```bash
# Start obstacle detection with default webcam
python main.py
```

- The application opens a video window titled **"Object Detection"**.
- When an obstacle is detected, you will hear **"Object Detected!"** via your speakers.
- Detected objects are highlighted with **green bounding boxes** on the video feed.
- Press **`Q`** to quit the application.

---

## 🔬 How It Works

```
Camera Feed → HSV Conversion → Color Thresholding → Contour Detection → Alert
```

1. **Video Capture** — Frames are captured from the webcam using OpenCV.
2. **HSV Conversion** — Each frame is converted from BGR to HSV color space for robust color-based detection.
3. **Thresholding** — A binary mask is generated using predefined HSV bounds to isolate potential obstacles.
4. **Contour Detection** — Contours are extracted from the mask to identify object boundaries.
5. **Alert System** — If contours are found, the pyttsx3 engine speaks an audio alert and bounding boxes are drawn on the frame.

---

## 🗂️ Project Structure

```
safestep/
├── main.py                 # Core detection and alert logic
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
├── LICENSE                 # MIT License
└── README.md               # Project documentation
```

---

## 🔮 Future Enhancements

- [ ] **Directional Audio** — Indicate obstacle position (left, right, center) with spatial audio cues
- [ ] **Deep Learning Model** — Replace HSV thresholding with MobileNet SSD or YOLOv8 for robust multi-class object detection
- [ ] **Distance Estimation** — Approximate obstacle distance using object size in frame
- [ ] **Multilingual Voice** — Support for alerts in multiple languages
- [ ] **Raspberry Pi Deployment** — Optimized build for portable, battery-powered hardware
- [ ] **Alert Cooldown** — Prevent repeated alerts for the same object within a short time window

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [OpenCV](https://opencv.org/) — Open Source Computer Vision Library
- [pyttsx3](https://pyttsx3.readthedocs.io/) — Offline Text-to-Speech Library
- [NumPy](https://numpy.org/) — Fundamental Package for Scientific Computing

---

<p align="center">
  Built with ❤️ to empower independence for visually impaired individuals
</p>
