import cv2
import numpy as np
import os
import requests
from moviepy.editor import ImageSequenceClip, VideoFileClip, concatenate_videoclips, CompositeVideoClip, ColorClip
from bs4 import BeautifulSoup

# Try importing Tesseract for OCR (Mode 2 Reading)
try:
    import pytesseract
    # WINDOWS USERS: If Tesseract is not in your PATH, uncomment and set the path below:
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except ImportError:
    pytesseract = None

class VideoEngine:
    def __init__(self):
        # Define paths
        self.asset_dir = os.path.join("assets", "images")
        self.video_cache = os.path.join("assets", "video_cache")
        self.temp_dir = os.path.join("assets", "temp")

        # Ensure directories exist
        os.makedirs(self.asset_dir, exist_ok=True)
        os.makedirs(self.video_cache, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

    def get_image_path(self, char):
        """Returns path for a character image (A-Z) for fallback."""
        char = char.lower()
        for ext in [".png", ".jpg", ".jpeg"]:
            path = os.path.join(self.asset_dir, char + ext)
            if os.path.exists(path):
                return path
        return None

    # --- ROBUST WEB SCRAPER ---
    def fetch_online_sign(self, word):
        """Scrapes SignASL.org for a sign video."""
        clean_word = "".join(filter(str.isalpha, word.lower()))
        if not clean_word: return None

        cache_path = os.path.join(self.video_cache, f"{clean_word}.mp4")

        # Check Cache
        if os.path.exists(cache_path):
            if os.path.getsize(cache_path) > 1024:
                print(f"📦 Found cached video for '{clean_word}'")
                return cache_path
            else:
                os.remove(cache_path)

        # Search Web
        url = f"https://www.signasl.org/sign/{clean_word}"
        headers = {"User-Agent": "Mozilla/5.0"}

        try:
            print(f"🔍 Searching SignASL.org for '{clean_word}'...")
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200: return None

            soup = BeautifulSoup(response.content, 'html.parser')
            video_url = None
            all_videos = soup.find_all('video')
            for vid in all_videos:
                src = vid.get('src')
                if not src and vid.find('source'):
                    src = vid.find('source').get('src')

                if src and (".mp4" in src or ".webm" in src):
                    video_url = src
                    break

            if video_url:
                if video_url.startswith("/"): video_url = "https://www.signasl.org" + video_url
                print(f"⬇️ Downloading: {video_url}")
                vid_data = requests.get(video_url, headers=headers, stream=True).content
                with open(cache_path, 'wb') as f:
                    f.write(vid_data)
                return cache_path

        except Exception as e:
            print(f"⚠️ Scraper Error: {e}")
            return None

        return None

    # --- HELPER: TEXT WRAPPING ---
    def _wrap_text(self, text, font, font_scale, thickness, max_width):
        words = text.split(' ')
        lines = []
        current_line = []

        for word in words:
            current_line.append(word)
            test_line = ' '.join(current_line)
            (width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            if width > max_width and len(current_line) > 1:
                current_line.pop()
                lines.append(' '.join(current_line))
                current_line = [word]
        lines.append(' '.join(current_line))
        return lines

    # --- HELPER: BINARY CONVERSION ---
    def _text_to_bits(self, text):
        # Convert text to binary string, padding each char to 8 bits
        bits = bin(int.from_bytes(text.encode('utf-8'), 'big'))[2:]
        return bits.zfill(8 * ((len(bits) + 7) // 8))

    # --- MODE 2: APPEND VISIBLE SLIDE ---
    def _append_visible_slide(self, input_path, output_path, text):
        print(f"[Mode 2] Appending visible slide...")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened(): return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # FIX: Use 'avc1' (H.264) for Browser Compatibility
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
        except:
            print("Warning: H.264 (avc1) not found, falling back to mp4v.")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Copy existing frames
        while True:
            ret, frame = cap.read()
            if not ret: break
            out.write(frame)

        # Create Text Slide (Black Background)
        slide = np.zeros((height, width, 3), dtype='uint8')
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1 if len(text) < 200 else 0.6
        thickness = 2
        margin = 50

        lines = self._wrap_text(text, font, scale, thickness, width - 2 * margin)
        line_height = 40
        y = (height - (len(lines) * line_height)) // 2

        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, font, scale, thickness)
            x = (width - tw) // 2
            cv2.putText(slide, line, (x, y + line_height), font, scale, (255, 255, 255), thickness)
            y += line_height

        # Append for 3 seconds
        for _ in range(int(fps * 3)):
            out.write(slide)

        cap.release()
        out.release()
        return True

    # --- MODE 1: EMBED HIDDEN TEXT (FIXED) ---
    def _embed_hidden_data(self, input_path, output_path, secret_text):
        print(f"[Mode 1] Embedding hidden data in last frame...")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened(): return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # FIX: Use 'avc1' (H.264) for Browser Compatibility
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
        except:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Prepare binary data (Null terminated)
        binary_secret = self._text_to_bits(secret_text) + "00000000"
        data_len = len(binary_secret)

        curr = 0
        while True:
            ret, frame = cap.read()
            if not ret: break

            # Embed in the LAST frame only
            if curr == total_frames - 1:
                flat = frame.flatten()

                if data_len < len(flat):
                    for i in range(data_len):
                        # FIX: Use 254 (0xFE) instead of ~1 to prevent uint8 overflow (-2)
                        # This clears the LSB safely
                        flat[i] = (flat[i] & 254) | int(binary_secret[i])

                    frame = flat.reshape((height, width, 3))
                    print(" -> Data encoded successfully.")
                else:
                    print("Error: Text too long for this video resolution.")

            out.write(frame)
            curr += 1

        cap.release()
        out.release()
        return True

    # --- MAIN GENERATOR ---
    def generate_sequence(self, text, output_path, force_spelling=False):
        """
        Generates sign language video, appends visible text (Mode 2),
        and hides steganography text (Mode 1).
        """
        words = text.split()
        clips = []

        print(f"🎬 Processing: {text} | Spelling Only: {force_spelling}")

        # 1. GENERATE BASE SIGN LANGUAGE VIDEO
        spacer = ColorClip(size=(1280, 720), color=(0, 0, 0), duration=1.0).set_fps(24)

        for i, word in enumerate(words):
            clean_word = "".join(filter(str.isalpha, word.lower()))
            if not clean_word: continue

            success = False
            # Try Web
            if not force_spelling:
                video_path = self.fetch_online_sign(clean_word)
                if video_path:
                    try:
                        clip = VideoFileClip(video_path)
                        if clip.duration < 0.1: raise ValueError("Short clip")

                        clip_resized = clip.resize(height=720)
                        if clip_resized.w % 2 != 0: clip_resized = clip_resized.resize(width=clip_resized.w - 1)
                        clip_resized = clip_resized.set_fps(24)

                        final_piece = CompositeVideoClip([clip_resized.set_position("center")], size=(1280, 720))
                        final_piece.duration = clip.duration
                        clips.append(final_piece)
                        success = True
                        print(f"✅ Added Clip: {clean_word}")
                    except:
                        if os.path.exists(video_path): os.remove(video_path)

            # Try Finger Spelling
            if not success:
                print(f"🔤 Finger-spelling: {clean_word}")
                clips.append(self._create_finger_spell_clip(clean_word))

            # Add Space
            if i < len(words) - 1:
                clips.append(spacer)

        if not clips: return None

        # Temp paths for intermediate steps
        temp_base = os.path.join(self.temp_dir, "temp_base.mp4")
        temp_mode2 = os.path.join(self.temp_dir, "temp_mode2.mp4")

        try:
            print(f"🔗 Stitching clips...")
            final_clip = concatenate_videoclips(clips, method="compose")

            # Write base video using MoviePy (Standard H264)
            final_clip.write_videofile(temp_base, fps=24, codec="libx264", audio=False, logger=None)

            # 2. APPLY MODE 2 (Visible Text Slide)
            if not self._append_visible_slide(temp_base, temp_mode2, text):
                print("❌ Mode 2 Failed")
                return temp_base

            # 3. APPLY MODE 1 (Hidden Text Encoding)
            if self._embed_hidden_data(temp_mode2, output_path, text):
                print(f"✅ FINAL VIDEO GENERATED: {output_path}")
            else:
                print("❌ Mode 1 Failed, returning Mode 2 output.")
                if os.path.exists(output_path): os.remove(output_path)
                os.rename(temp_mode2, output_path)

            # Cleanup Temp Files
            if os.path.exists(temp_base): os.remove(temp_base)
            if os.path.exists(temp_mode2): os.remove(temp_mode2)

            return output_path

        except Exception as e:
            print(f"❌ Generation Error: {e}")
            return None

    def _create_finger_spell_clip(self, word):
        """Creates video from static letter images."""
        frames = []
        for char in word:
            p = self.get_image_path(char)
            if p:
                img = cv2.imread(p)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w, _ = img.shape
                    scale = min(720 / h, 1280 / w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    img = cv2.resize(img, (new_w, new_h))

                    bg = np.zeros((720, 1280, 3), dtype=np.uint8)
                    y_off = (720 - new_h) // 2
                    x_off = (1280 - new_w) // 2
                    bg[y_off:y_off + new_h, x_off:x_off + new_w] = img

                    frames.extend([bg] * 24)

        if not frames:
            err = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(err, f"MISSING: {word}", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            frames = [err] * 24

        return ImageSequenceClip(frames, fps=24)


# ==========================================
# NEW DECODER CLASS FOR MODE 3
# ==========================================
class VideoDecoder:
    def decode_mode1_hidden(self, video_path):
        """Extracts hidden LSB binary data from the last frame."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return "Error: Cannot open video."

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, frame = cap.read()
        cap.release()

        if not ret: return "Error: Could not read last frame."

        print("Searching for hidden binary data...")
        img_data = frame.flatten()
        binary_data = ""

        # Limit check to first 10,000 bits to avoid hanging on huge videos
        max_bits = 10000
        for i in range(min(len(img_data), max_bits)):
            binary_data += str(img_data[i] & 1)

        decoded_text = ""
        for i in range(0, len(binary_data), 8):
            byte = binary_data[i:i+8]
            if len(byte) < 8: break
            char_code = int(byte, 2)
            if char_code == 0: break # Null terminator found
            decoded_text += chr(char_code)

        return decoded_text if decoded_text else "No hidden content found."

    def decode_mode2_visible(self, video_path):
        """Uses OCR to read visible text from the last slide."""
        if pytesseract is None:
            return "Error: 'pytesseract' library not installed or found."

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return "Error: Cannot open video."

        # Go to last frame
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Back up 5 frames to avoid end-of-video glitches
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 5))
        ret, frame = cap.read()
        cap.release()

        if not ret: return "Error: Could not read frame."

        # Image Pre-processing for OCR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        try:
            text = pytesseract.image_to_string(thresh)
            return text.strip()
        except Exception as e:
            return f"OCR Error: {e}"