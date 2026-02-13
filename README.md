
# 🤟 ASL Multimedia System  
**Multimodal American Sign Language Translation & Learning Platform**

An interactive multimedia system for **bidirectional American Sign Language (ASL) translation and learning**, featuring text-to-video generation, video-to-text recognition, webcam-based detection, OCR fallback, and gamified quiz modes — all delivered through a Streamlit web interface.

---

## 🚀 Features

- 🔤 Text → Sign Video (Character-by-Character)
- 🧩 Text → Sign Video (Word-by-Word with Character Fallback)
- 🎥 Video → Text Recognition
- 📷 Webcam-based Sign Detection
- 🔎 OCR-based Text Extraction from Video (Tesseract)
- 🌐 Automatic Sign Video Fetching (Web Scraper fallback)
- 🧠 Quiz & Learning Modes
- 👤 Role-based Modes
- ⚡ Real-time Processing with MediaPipe & OpenCV
- 🎬 MP4-based realistic sign rendering

---

## 🏗️ Tech Stack

| Layer | Technology |
|--------|-------------|
Frontend UI | Streamlit |
Computer Vision | OpenCV, MediaPipe |
Video Processing | MoviePy |
OCR | Tesseract OCR + pytesseract |
Data Handling | NumPy, Pandas, SciPy |
Audio | gTTS |
Web Scraping | Requests, BeautifulSoup |
Language | Python 3.9+ |

---

## 📂 Project Structure

ASL-Multimedia-System/
│
├── app.py
├── requirements.txt
│
├── assets/
│   ├── images/
│   ├── temp/
│   └── video_cache/
│
├── modules/
├── utils/
└── README.md

---

# 🛠️ Installation & Setup

## ✅ Prerequisites

- Python 3.9+
- Webcam (for Translator & Quiz modes)
- Tesseract OCR

---

## 📥 Clone Repository

```bash
git clone https://github.com/SuyashGupta04/ASL-Multimedida-System.git
cd ASL-Multimedida-System
```

---

## 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

macOS/Linux OpenCV fix if needed:

```bash
pip install opencv-python-headless
```

---

## 🔍 Install Tesseract OCR

### Windows
Install to:
C:\Program Files\Tesseract-OCR  
Add to PATH.

### macOS
```bash
brew install tesseract
```

### Linux
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

---

## 🎞️ Asset Setup

Create:

assets/images/
assets/temp/
assets/video_cache/

Place sign videos (.mp4) inside assets/images/

Missing signs are auto-downloaded by the built-in scraper.

---

## ▶️ Run Application

```bash
streamlit run app.py
```

---

## 🎮 System Modes

Mode 1 — Character-based Text → Video  
Mode 2 — Word-based Text → Video with fallback  
Mode 3 — Video → Text  
Mode 4 — Webcam Detection  
Mode 5 — Learning Mode  
Mode 6 — Quiz Mode  

---

## 📄 requirements.txt

```txt
streamlit
opencv-python
mediapipe
moviepy==1.0.3
numpy
pandas
scipy
gtts
requests
beautifulsoup4
pytesseract
watchdog
```

---

## ⚡ Performance Notes

- First run may be slower due to downloads and caching
- Cached videos stored in video_cache/
- Good lighting improves detection accuracy

---

## 🔐 Limitations

- Accuracy depends on lighting and camera quality
- Word-level signs limited by dataset
- OCR depends on video clarity

---

## 🤝 Contributing

Fork → Branch → Commit → Push → Pull Request

Include test results and screenshots when applicable.

---

## 📌 Roadmap

- Deep learning sign classifier
- Multi-language support
- Mobile deployment
- Custom dataset training

---

## 📜 License

Add your preferred license (MIT recommended).

---

## 👨‍💻 Author

Suyash Gupta  
ASL Multimedia System Project
