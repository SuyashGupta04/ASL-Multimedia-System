🛠️ Installation Guide
1. Prerequisites
Python 3.8 or higher.

A working webcam (for live translation).

2. Clone the Repository
Bash
git clone https://github.com/SuyashGupta04/ASL-Multimedia-System/
cd ASL-Multimedia-System
3. Create a Virtual Environment (Recommended)


Bash
python -m venv venv
venv\Scripts\activate
Mac/Linux:

Bash
python3 -m venv venv
source venv/bin/activate
4. Install Dependencies

Bash
pip install streamlit opencv-python mediapipe moviepy requests beautifulsoup4 pandas matplotlib python-pptx numpy
📂 Asset Setup (Crucial!)
The system requires a set of static images for Finger-Spelling (A-Z).

Create a folder named assets in the main directory.

Inside assets, create a folder named images.

Add images: Place images named a.png, b.png, ... z.png (or .jpg) into assets/images.

Note: Without these images, the Finger-Spelling fallback mode will show "Missing Asset" errors.

Folder Structure:

Plaintext
ASL-Multimedia-System/
├── app.py
├── engines/
│   ├── image_engine.py
│   ├── video_engine.py
│   └── translator_engine.py
├── utils/
│   ├── auth.py
│   └── feedback.py
├── assets/
│   ├── images/       <-- Put a.png, b.png here
│   └── video_cache/  <-- Auto-created by app
├── users.json        <-- Auto-created
├── feedback.json     <-- Auto-created
└── README.md
▶️ How to Run
Open your terminal in the project folder.

Run the Streamlit application:

Bash
streamlit run app.py
The app will open automatically in your browser (usually at http://localhost:8501).

📖 Usage Guide
1. Login / Register
Register: Go to the "Register" tab to create a new account.

Login: Use your new credentials.

Admin Access: To access Admin features, manually edit users.json and change a user's role from "user" to "admin", or register a user named admin (if logic permits).

2. Text-to-Sign (Tab 1 & 2)
Tab 1 (Word Animation): Best for single words. Generates pure finger-spelling videos.

Tab 2 (Smart Stitcher): Best for sentences. It will download real videos from the web and stitch them.

Tip: The first time you run a sentence, it might take a moment to download videos. Subsequent runs use the cache and are instant.

3. Sign-to-Text (Tab 3)
Input: Choose "Live Webcam" or "Upload Video".

Smart Translation: Click "Run Smart Translation". The system auto-detects if the video is a static image (Pattern Match) or a real human (AI Skeleton).

Spaces: To create a space between words in real-time, simply move your hand out of the frame or show a black screen for ~0.5 seconds.

4. Admin Dashboard
Log in as an Admin to see the Research Lab (for benchmarking algorithms) and the Feedback Dashboard (to view user ratings).

🧩 Troubleshooting
MediaPipe Error on Mac: If you see an error related to mediapipe, try uninstalling it and installing mediapipe-silicon (for M1/M2 chips) or ensure you are using a compatible Python version (3.8-3.10 often works best).

Video Not Saving: Ensure the temp_output folder exists (the app usually creates it automatically).

Scraper Fails: If the web scraper fails, check your internet connection. The system will automatically fall back to finger-spelling images.
