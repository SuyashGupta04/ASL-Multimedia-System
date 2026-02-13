import cv2
import numpy as np
import os
import requests
from bs4 import BeautifulSoup

# Try importing Tesseract
try:
    import pytesseract
except ImportError:
    pytesseract = None


class VideoEngine:
    def __init__(self):
        # Define paths
        self.asset_dir = os.path.join("assets", "images")
        self.video_cache = os.path.join("assets", "video_cache")
        self.temp_dir = os.path.join("assets", "temp")

        os.makedirs(self.asset_dir, exist_ok=True)
        os.makedirs(self.video_cache, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        # Standard Resolution (HD 720p)
        self.frame_width = 1280
        self.frame_height = 720
        self.fps = 30

    def get_sign_asset(self, char):
        char = char.lower()
        vid_path = os.path.join(self.asset_dir, f"{char}.mp4")
        if os.path.exists(vid_path): return vid_path
        for ext in [".jpg", ".png", ".jpeg"]:
            img_path = os.path.join(self.asset_dir, f"{char}{ext}")
            if os.path.exists(img_path): return img_path
        return None

    def get_image_path(self, char):
        return self.get_sign_asset(char)

    # --- ROBUST WEB SCRAPER ---
    def fetch_online_sign(self, word):
        clean_word = "".join(filter(str.isalpha, word.lower()))
        if not clean_word: return None
        cache_path = os.path.join(self.video_cache, f"{clean_word}.mp4")

        if os.path.exists(cache_path):
            if os.path.getsize(cache_path) > 1024:
                return cache_path
            else:
                os.remove(cache_path)

        url = f"https://www.signasl.org/sign/{clean_word}"
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code != 200: return None
            soup = BeautifulSoup(response.content, 'html.parser')
            video_url = None
            for vid in soup.find_all('video'):
                src = vid.get('src') or (vid.find('source').get('src') if vid.find('source') else None)
                if src and (".mp4" in src or ".webm" in src):
                    video_url = src
                    break
            if video_url:
                if video_url.startswith("/"): video_url = "https://www.signasl.org" + video_url
                vid_data = requests.get(video_url, headers=headers, stream=True).content
                with open(cache_path, 'wb') as f:
                    f.write(vid_data)
                return cache_path
        except:
            return None
        return None

    # --- MAIN GENERATOR ---
    def generate_sequence(self, text, output_path, force_spelling=False, speed=1.0):
        words = text.split()

        # Initialize Video Writer
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.frame_width, self.frame_height))

        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.frame_width, self.frame_height))

        # Spacer Frame
        spacer_frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

        for i, word in enumerate(words):
            clean_word = "".join(filter(str.isalpha, word.lower()))
            if not clean_word: continue

            asset_path = None

            # 1. Try to fetch whole word
            if not force_spelling:
                asset_path = self.fetch_online_sign(clean_word)

            # 2. Write Video (Apply Speed)
            if asset_path:
                self._write_video_to_stream(out, asset_path, speed)

            # 3. Finger Spelling (Apply Speed)
            else:
                for char in clean_word:
                    char_asset = self.get_sign_asset(char)
                    if char_asset:
                        if char_asset.endswith(".mp4"):
                            self._write_video_to_stream(out, char_asset, speed)
                        else:
                            self._write_image_to_stream(out, char_asset, duration_sec=1.0 / speed)
                    else:
                        self._write_error_frame(out, char, duration_sec=1.0 / speed)

            # 4. Add Space
            if i < len(words) - 1:
                spacer_duration = int((self.fps * 0.5) / speed)
                for _ in range(spacer_duration):
                    out.write(spacer_frame)

        # 5. --- NEW: APPEND TEXT SLIDE FOR MODE 3 ---
        # This writes the full input text at the end for 2 seconds
        self._write_text_slide(out, text, duration_sec=2.0)

        out.release()
        return output_path

    # --- STREAMING HELPERS ---
    def _write_video_to_stream(self, out, video_path, speed=1.0):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return

        total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_src_frames == 0: return

        src_pos = 0.0

        while True:
            target_src_idx = int(src_pos)
            current_cap_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            while current_cap_pos < target_src_idx:
                cap.read()
                current_cap_pos += 1

            ret, frame = cap.read()
            if not ret: break

            resized = cv2.resize(frame, (self.frame_width, self.frame_height))
            out.write(resized)

            src_pos += speed
            if src_pos >= total_src_frames: break

        cap.release()

    def _write_image_to_stream(self, out, img_path, duration_sec=1.0):
        img = cv2.imread(img_path)
        if img is None: return
        resized = cv2.resize(img, (self.frame_width, self.frame_height))
        for _ in range(int(self.fps * duration_sec)):
            out.write(resized)

    def _write_error_frame(self, out, label, duration_sec=1.0):
        frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        frame[:] = (0, 0, 100)
        cv2.putText(frame, f"MISSING: {label.upper()}", (100, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        for _ in range(int(self.fps * duration_sec)):
            out.write(frame)

    # --- NEW: TEXT SLIDE GENERATOR ---
    def _write_text_slide(self, out, text, duration_sec=2.0):
        """Creates a black frame with white text for OCR detection."""
        frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        thickness = 3
        color = (255, 255, 255)  # White

        # Calculate text size to center it
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (self.frame_width - text_size[0]) // 2
        text_y = (self.frame_height + text_size[1]) // 2

        # Ensure text fits on screen (Simple shrink if too wide)
        if text_x < 10:
            font_scale = 1.0
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (self.frame_width - text_size[0]) // 2

        cv2.putText(frame, text.upper(), (text_x, text_y), font, font_scale, color, thickness)

        # Write frames
        for _ in range(int(self.fps * duration_sec)):
            out.write(frame)


class VideoDecoder:
    def decode_mode2_visible(self, video_path):
        if pytesseract is None: return "Error: OCR Library missing"
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Check the last 1 second of video where the text slide is
        # We start checking from frame_count - 30 to end
        search_start = max(0, frame_count - 45)
        cap.set(cv2.CAP_PROP_POS_FRAMES, search_start)

        detected_text = ""

        # Scan last few frames to find best text
        for _ in range(45):
            ret, frame = cap.read()
            if not ret: break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

            try:
                txt = pytesseract.image_to_string(thresh).strip()
                if len(txt) > len(detected_text):  # Keep longest detection
                    detected_text = txt
            except:
                pass

        cap.release()
        return detected_text if detected_text else "OCR Failed: No text found."