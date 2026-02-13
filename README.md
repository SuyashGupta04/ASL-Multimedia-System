## 🛠️ Installation & Setup

Get the system up and running in less than 5 minutes.

### 📋 Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python 3.9+** ([Download Here](https://www.python.org/downloads/))
2.  **Git** ([Download Here](https://git-scm.com/downloads))
3.  **Tesseract-OCR Engine** (Required for the Mode 3 Decoder)

    > **⚠️ Critical Step:** You *must* install Tesseract for the "Visible Words" decoder to work.
    > * **Windows:** [Download Installer](https://github.com/UB-Mannheim/tesseract/wiki).
    >     * *Note:* During installation, copy the installation path (default is `C:\Program Files\Tesseract-OCR`).
    > * **macOS:** `brew install tesseract`
    > * **Linux (Ubuntu/Debian):** `sudo apt-get install tesseract-ocr`

---

### 🚀 Step-by-Step Installation

#### 1. Clone the Repository
Open your terminal or command prompt and run:
```bash ```
git clone [https://github.com/your-username/asl-multimedia-system.git](https://github.com/SuyashGupta04/ASL-Multimedia-System.git)
cd ASL-Multimedia-System

2. Create a Virtual Environment (Recommended)This keeps your project dependencies isolated to avoid conflicts.Windows:
Bash
python -m venv venv
.\venv\Scripts\activate
macOS / Linux:Bash
python3 -m venv venv
source venv/bin/activate
3. Install Python DependenciesInstall all required libraries (OpenCV, MediaPipe, Streamlit, etc.):
Bash
pip install --upgrade pip
pip install -r requirements.txt
4. Configure Folders (First Run Only)Ensure the asset directories exist to prevent "File Not Found" errors.
Mac/Linux:Bash
mkdir -p assets/images assets/video_cache assets/temp
Windows (PowerShell):PowerShell
md assets/images, assets/video_cache, assets/temp
5. Run the ApplicationLaunch the web interface:
Bash
streamlit run app.py
The app should automatically open in your default browser at http://localhost:8501.🔧 Common TroubleshootingError MessageSolutionTesseractNotFound ErrorWindows Users: Open engines/video_engine.py and uncomment the line:  pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'ModuleNotFoundError: No module named '...'Run pip install -r requirements.txt again to ensure nothing was missed.ffmpeg not foundIf video generation fails, run: pip install imageio-ffmpeg
