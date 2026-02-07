import cv2
import numpy as np
import os
import requests
from moviepy.editor import ImageSequenceClip, VideoFileClip, concatenate_videoclips, CompositeVideoClip, ColorClip
from bs4 import BeautifulSoup


class VideoEngine:
    def __init__(self):
        # Define paths
        self.asset_dir = os.path.join("assets", "images")
        self.video_cache = os.path.join("assets", "video_cache")

        # Ensure directories exist
        os.makedirs(self.asset_dir, exist_ok=True)
        os.makedirs(self.video_cache, exist_ok=True)

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

            # Find video
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

    # --- MAIN VIDEO GENERATOR ---
    def generate_sequence(self, text, output_path, force_spelling=False):
        """
        Generates video sequence with LONG SPACER clips between words.
        """
        words = text.split()
        clips = []

        print(f"🎬 Processing: {text} | Spelling Only: {force_spelling}")

        # --- FIX: INCREASE SPACER TO 1.0 SECOND ---
        # This ensures the translator definitely sees the "Black Screen" gap
        spacer = ColorClip(size=(1280, 720), color=(0, 0, 0), duration=1.0).set_fps(24)

        for i, word in enumerate(words):
            clean_word = "".join(filter(str.isalpha, word.lower()))
            if not clean_word: continue

            success = False

            # 1. Try Web Video (ONLY if NOT forcing spelling)
            if not force_spelling:
                video_path = self.fetch_online_sign(clean_word)
                if video_path:
                    try:
                        clip = VideoFileClip(video_path)
                        if clip.duration < 0.1: raise ValueError("Short clip")

                        # Resize to 720p
                        clip_resized = clip.resize(height=720)
                        if clip_resized.w % 2 != 0:
                            clip_resized = clip_resized.resize(width=clip_resized.w - 1)
                        clip_resized = clip_resized.set_fps(24)

                        final_piece = CompositeVideoClip([clip_resized.set_position("center")], size=(1280, 720))
                        final_piece.duration = clip.duration
                        clips.append(final_piece)
                        success = True
                        print(f"✅ Added Clip: {clean_word}")
                    except:
                        if os.path.exists(video_path): os.remove(video_path)

            # 2. Fallback to Finger Spelling
            if not success:
                print(f"🔤 Finger-spelling: {clean_word}")
                clips.append(self._create_finger_spell_clip(clean_word))

            # 3. INSERT SPACE (Unless it's the last word)
            if i < len(words) - 1:
                clips.append(spacer)
                print("➕ Added Space (1.0s)")

        if not clips: return None

        try:
            print(f"🔗 Stitching {len(clips)} clips...")
            final_clip = concatenate_videoclips(clips, method="compose")
            final_clip.write_videofile(output_path, fps=24, codec="libx264", audio=False, logger=None)
            return output_path
        except Exception as e:
            print(f"❌ Stitch Error: {e}")
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

                    # 1 Second per letter
                    frames.extend([bg] * 24)

        if not frames:
            err = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(err, f"MISSING: {word}", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            frames = [err] * 24

        return ImageSequenceClip(frames, fps=24)