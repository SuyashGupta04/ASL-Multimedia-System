import cv2
import numpy as np
import os
import requests
import time
import subprocess
from bs4 import BeautifulSoup

# --- Audio & Video Manipulation ---
try:
    from gtts import gTTS
    from moviepy.editor import AudioFileClip
except ImportError:
    gTTS = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

# Get the local FFmpeg executable path safely
try:
    from imageio_ffmpeg import get_ffmpeg_exe

    FFMPEG_PATH = get_ffmpeg_exe()
except ImportError:
    FFMPEG_PATH = "ffmpeg"


class VideoEngine:
    def __init__(self):
        self.asset_dir = os.path.join("assets", "images")
        self.video_cache = os.path.join("assets", "video_cache")
        self.temp_dir = os.path.join("assets", "temp")

        os.makedirs(self.asset_dir, exist_ok=True)
        os.makedirs(self.video_cache, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

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

    # --- PERFECT SYNC AUDIO GENERATORS ---
    def _make_audio_segment(self, text, target_dur_sec, ts, uid):
        """Generates TTS for a specific word/letter, stretches it, and normalizes it."""
        if target_dur_sec <= 0: return None
        raw_mp3 = os.path.join(self.temp_dir, f"raw_{ts}_{uid}.mp3")
        out_wav = os.path.join(self.temp_dir, f"seg_{ts}_{uid}.wav")

        try:
            # Uppercase helps gTTS spell single characters out loud
            tts_text = text.upper() if len(text) == 1 else text
            gTTS(text=tts_text, lang='en').save(raw_mp3)

            # Use MoviePy just to measure exact duration
            with AudioFileClip(raw_mp3) as clip:
                orig_dur = clip.duration

            factor = orig_dur / target_dur_sec if target_dur_sec > 0 else 1.0

            # Chain FFmpeg atempo filters if extreme stretch is needed
            atempo_filters = []
            f = factor
            while f > 2.0:
                atempo_filters.append("atempo=2.0")
                f /= 2.0
            while f < 0.5:
                atempo_filters.append("atempo=0.5")
                f /= 0.5
            atempo_filters.append(f"atempo={f}")
            filter_str = ",".join(atempo_filters)

            # Command forces 24000Hz Mono to match generated silence
            cmd = [
                FFMPEG_PATH, "-y", "-i", raw_mp3,
                "-filter_complex", f"[0:a]{filter_str}[a]",
                "-map", "[a]",
                "-ar", "24000", "-ac", "1",
                "-t", str(target_dur_sec),
                out_wav
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return out_wav
        except Exception as e:
            print(f"Segment Error: {e}")
            return self._make_silence(target_dur_sec, ts, uid)

    def _make_silence(self, dur_sec, ts, uid):
        """Generates pure silence at 24000Hz Mono to match the TTS audio perfectly."""
        if dur_sec <= 0: return None
        out_wav = os.path.join(self.temp_dir, f"silence_{ts}_{uid}.wav")

        cmd = [
            FFMPEG_PATH, "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
            "-t", str(dur_sec),
            "-ar", "24000", "-ac", "1",
            out_wav
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return out_wav

    # --- MAIN GENERATOR ---
    def generate_sequence(self, text, output_path, force_spelling=False, speed=1.0):
        words = text.split()
        ts = int(time.time())

        temp_vid_path = os.path.join(self.temp_dir, f"silent_video_{ts}.mp4")
        audio_segments = []
        files_to_cleanup = []

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(temp_vid_path, fourcc, self.fps, (self.frame_width, self.frame_height))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_vid_path, fourcc, self.fps, (self.frame_width, self.frame_height))

        spacer_frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

        print("🎬 Building Video and Segmented Audio Tracks...")

        # 1. BUILD VIDEO AND PARALLEL AUDIO TRACKS
        for i, word in enumerate(words):
            clean_word = "".join(filter(str.isalpha, word.lower()))
            if not clean_word: continue

            asset_path = None
            if not force_spelling:
                asset_path = self.fetch_online_sign(clean_word)

            # FULL WORD MODE
            if asset_path:
                frames = self._write_video_to_stream(out, asset_path, speed)
                dur = frames / self.fps
                if gTTS:
                    wav = self._make_audio_segment(clean_word, dur, ts, f"{i}_w")
                    if wav:
                        audio_segments.append(wav)
                        files_to_cleanup.append(wav)

            # SPELLING MODE (Fallback or Forced)
            else:
                for j, char in enumerate(clean_word):
                    char_asset = self.get_sign_asset(char)
                    frames = 0
                    if char_asset:
                        if char_asset.endswith(".mp4"):
                            frames = self._write_video_to_stream(out, char_asset, speed)
                        else:
                            frames = self._write_image_to_stream(out, char_asset, duration_sec=1.0 / speed)
                    else:
                        frames = self._write_error_frame(out, char, duration_sec=1.0 / speed)

                    dur = frames / self.fps
                    if gTTS:
                        wav = self._make_audio_segment(char, dur, ts, f"{i}_{j}_c")
                        if wav:
                            audio_segments.append(wav)
                            files_to_cleanup.append(wav)

            # ADD SPACE BETWEEN WORDS
            if i < len(words) - 1:
                spacer_frames = int((self.fps * 0.5) / speed)
                for _ in range(spacer_frames):
                    out.write(spacer_frame)

                dur = spacer_frames / self.fps
                if gTTS:
                    wav = self._make_silence(dur, ts, f"{i}_s")
                    if wav:
                        audio_segments.append(wav)
                        files_to_cleanup.append(wav)

        # 2. ADD OCR TEXT SLIDE
        self._write_text_slide(out, text, duration_sec=2.0)
        out.release()

        # Audio is silent during the OCR text slide
        if gTTS:
            wav = self._make_silence(2.0, ts, "end_slide")
            if wav:
                audio_segments.append(wav)
                files_to_cleanup.append(wav)

        time.sleep(0.5)

        # 3. MERGE EVERYTHING
        final_audio_path = os.path.join(self.temp_dir, f"final_audio_{ts}.m4a")
        audio_list_file = os.path.join(self.temp_dir, f"concat_list_{ts}.txt")
        files_to_cleanup.extend([temp_vid_path, final_audio_path, audio_list_file])

        try:
            if gTTS and audio_segments:
                print("🎙️ Stitching Audio Segments...")

                # Create FFmpeg concat list (Ensuring paths are safe)
                with open(audio_list_file, 'w') as f:
                    for seg in audio_segments:
                        safe_path = os.path.abspath(seg).replace('\\', '/')
                        f.write(f"file '{safe_path}'\n")

                # Concat all audio pieces
                cmd_concat = [
                    FFMPEG_PATH, "-y", "-f", "concat", "-safe", "0",
                    "-i", audio_list_file, "-c:a", "aac", final_audio_path
                ]

                res = subprocess.run(cmd_concat, capture_output=True, text=True)
                if res.returncode != 0:
                    raise Exception(f"Concat failed: {res.stderr}")

                print("🎬 Merging Final Output...")

                # Merge Video and Audio
                cmd_merge = [
                    FFMPEG_PATH, "-y",
                    "-i", temp_vid_path,
                    "-i", final_audio_path,
                    "-c:v", "copy",
                    "-c:a", "copy",
                    output_path
                ]
                res2 = subprocess.run(cmd_merge, capture_output=True, text=True)
                if res2.returncode != 0:
                    raise Exception(f"Merge failed: {res2.stderr}")

            else:
                os.rename(temp_vid_path, output_path)

        except Exception as e:
            print(f"\n❌ Pipeline Failed: {e}\n")
            if os.path.exists(output_path): os.remove(output_path)
            os.rename(temp_vid_path, output_path)

        finally:
            # Cleanup all temporary segments and files
            for f in files_to_cleanup:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass
            for f in os.listdir(self.temp_dir):
                if f.startswith(f"raw_{ts}"):
                    try:
                        os.remove(os.path.join(self.temp_dir, f))
                    except:
                        pass

        return output_path

    # --- STREAMING HELPERS ---
    def _write_video_to_stream(self, out, video_path, speed=1.0):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return 0
        total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_src_frames == 0: return 0

        frames_written = 0
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
            frames_written += 1

            src_pos += speed
            if src_pos >= total_src_frames: break
        cap.release()
        return frames_written

    def _write_image_to_stream(self, out, img_path, duration_sec=1.0):
        img = cv2.imread(img_path)
        if img is None: return 0
        resized = cv2.resize(img, (self.frame_width, self.frame_height))
        frames = int(self.fps * duration_sec)
        for _ in range(frames):
            out.write(resized)
        return frames

    def _write_error_frame(self, out, label, duration_sec=1.0):
        frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        frame[:] = (0, 0, 100)
        cv2.putText(frame, f"MISSING: {label.upper()}", (100, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        frames = int(self.fps * duration_sec)
        for _ in range(frames):
            out.write(frame)
        return frames

    def _write_text_slide(self, out, text, duration_sec=2.0):
        frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        thickness = 3
        color = (255, 255, 255)

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (self.frame_width - text_size[0]) // 2
        text_y = (self.frame_height + text_size[1]) // 2

        if text_x < 10:
            font_scale = 1.0
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (self.frame_width - text_size[0]) // 2

        cv2.putText(frame, text.upper(), (text_x, text_y), font, font_scale, color, thickness)
        for _ in range(int(self.fps * duration_sec)):
            out.write(frame)


class VideoDecoder:
    def decode_mode2_visible(self, video_path):
        if pytesseract is None: return "Error: OCR Library missing"
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        search_start = max(0, frame_count - 45)
        cap.set(cv2.CAP_PROP_POS_FRAMES, search_start)
        detected_text = ""

        for _ in range(45):
            ret, frame = cap.read()
            if not ret: break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

            try:
                txt = pytesseract.image_to_string(thresh).strip()
                if len(txt) > len(detected_text): detected_text = txt
            except:
                pass

        cap.release()
        return detected_text if detected_text else "OCR Failed: No text found."