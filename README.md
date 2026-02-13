🛠️ Installation & Setup
Follow these steps to set up the ASL Multimedia System on your local machine.

1. Prerequisites
Python 3.9+ (Tested on 3.9 and 3.10)

Webcam (For Translator & Quiz modes)

Tesseract OCR (Required for Mode 3 & Translator Fallback)

2. Clone the Repository
Bash
git clone https://github.com/your-username/asl-research-project.git
cd asl-research-project
3. Install Python Dependencies
We use streamlit, opencv, mediapipe, and moviepy for the core engine.

Bash
pip install -r requirements.txt
Note for Mac/Linux Users: If you encounter errors with opencv, try: pip install opencv-python-headless

4. Install Tesseract OCR (Critical)
This project uses Tesseract for reading text from videos (Mode 3).

Windows:

Download the installer from UB-Mannheim/tesseract.

Install it to C:\Program Files\Tesseract-OCR.

Add this path to your System Environment Variables if needed.

macOS (Homebrew):

Bash
brew install tesseract
Linux (Ubuntu/Debian):

Bash
sudo apt-get update
sudo apt-get install tesseract-ocr
5. Asset Setup (Video Library)
This system uses MP4 video assets for realistic sign language generation.

Create the folder structure:

/assets
    /images      <-- Place your .mp4 files here (a.mp4, b.mp4...)
    /temp        <-- Auto-generated
    /video_cache <-- Auto-generated
Note: If you don't have local assets, the system's Web Scraper will automatically download missing signs from SignASL.org during generation.

6. Run the Application
Launch the web interface using Streamlit:

Bash
streamlit run app.py
📦 requirements.txt
Create a file named requirements.txt with the following content:

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
