import cv2
import numpy as np
import os
import subprocess
import mediapipe as mp


class TranslatorEngine:
    def __init__(self):
        # Initialize MediaPipe (For Human Hands)
        if not hasattr(mp, 'solutions'):
            raise ImportError("MediaPipe installed but 'solutions' missing.")
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        # Load Reference Images (For Static Letters)
        self.refs = self._load_references("assets/images")

    def _load_references(self, asset_dir):
        """Pre-loads A-Z images for pattern matching."""
        refs = {}
        if not os.path.exists(asset_dir): return refs

        for char in "abcdefghijklmnopqrstuvwxyz":
            for ext in [".png", ".jpg", ".jpeg"]:
                p = os.path.join(asset_dir, char + ext)
                if os.path.exists(p):
                    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        refs[char.upper()] = cv2.resize(img, (100, 100))
                    break
        return refs

    # --- CORE ALGORITHM ---
    def analyze_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 0. CHECK FOR BLACK SPACER
        avg_brightness = np.mean(gray)
        if avg_brightness < 40:  # Relaxed threshold
            return "SPACE", 1.0, "SPACER"

        # 1. PATTERN MATCHING
        h, w = gray.shape
        center_crop = gray[int(h * 0.1):int(h * 0.9), int(w * 0.2):int(w * 0.8)]
        if center_crop.size == 0: center_crop = gray

        target = cv2.resize(center_crop, (100, 100))
        best_char = "?"
        max_score = 0.0

        for char, ref in self.refs.items():
            res = cv2.matchTemplate(target, ref, cv2.TM_CCOEFF_NORMED)
            score = res[0][0]
            if score > max_score:
                max_score = score
                best_char = char

        if max_score > 0.60:
            return best_char, max_score, "PATTERN-MATCH"

        # 2. MEDIAPIPE
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            # Simple Geometry Checks
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                index_open = lm[8].y < lm[6].y
                middle_open = lm[12].y < lm[10].y
                ring_open = lm[16].y < lm[14].y
                pinky_open = lm[20].y < lm[18].y

                if index_open and not middle_open and not ring_open and not pinky_open:
                    return "D / 1", 0.8, "AI-SKELETON"
                elif index_open and middle_open and not ring_open and not pinky_open:
                    return "V / 2", 0.8, "AI-SKELETON"
                elif index_open and middle_open and ring_open and not pinky_open:
                    return "W / 6", 0.8, "AI-SKELETON"
                elif index_open and middle_open and ring_open and pinky_open:
                    return "B / 5", 0.8, "AI-SKELETON"
                elif not index_open and not middle_open and not ring_open and not pinky_open:
                    return "S / A", 0.8, "AI-SKELETON"
                elif not index_open and not middle_open and not ring_open and pinky_open:
                    return "I", 0.8, "AI-SKELETON"
                elif index_open and not middle_open and not ring_open and pinky_open:
                    return "Y", 0.8, "AI-SKELETON"

            return "[SIGN]", 0.5, "AI-SKELETON"

        return "?", 0.0, "NONE"

    # --- VIDEO PROCESSOR ---
    def process_video_smart(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None, "Error opening video"

        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24

        out_path = video_path.replace(".mp4", "_analyzed.mp4")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        detected_text_blocks = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            text, conf, method = self.analyze_frame(frame)

            # Group consecutive detections
            if not detected_text_blocks:
                detected_text_blocks.append({'text': text, 'count': 1})
            else:
                last = detected_text_blocks[-1]
                if last['text'] == text:
                    last['count'] += 1
                else:
                    detected_text_blocks.append({'text': text, 'count': 1})

            # Draw HUD
            cv2.rectangle(frame, (0, 0), (width, 80), (0, 0, 0), -1)
            color = (0, 255, 0)
            if method == "AI-SKELETON": color = (255, 200, 0)
            if text == "SPACE" or text == "?": color = (100, 100, 100)

            cv2.putText(frame, f"DETECTED: {text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            if "AI" in method:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for lm in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)

            out.write(frame)

        cap.release()
        out.release()

        # --- CONSTRUCT FINAL SENTENCE ---
        final_sentence = []

        for block in detected_text_blocks:
            t = block['text']
            count = block['count']

            # --- CRITICAL FIX: TREAT '?' AS SPACE ---
            # If nothing detected or explicit SPACE for > 10 frames (~0.4s) -> INSERT SPACE
            if (t == "SPACE" or t == "?") and count > 10:
                if not final_sentence or final_sentence[-1] != " ":
                    final_sentence.append(" ")
                continue

            # Filter short noise for letters
            if count < 8: continue
            if t == "?" or t == "SPACE": continue  # Skip short gaps

            clean_t = t.split(" / ")[0]

            # Calculate duration: 1 sec = 1 char
            seconds = count / fps
            repeats = int(round(seconds))
            if repeats < 1: repeats = 1

            final_sentence.append(clean_t * repeats)

        final_str = "".join(final_sentence)
        final_str = " ".join(final_str.split())  # Clean spaces

        return out_path, final_str

    # --- LIVE WEBCAM HELPER ---
    def process_frame(self, frame):
        text, conf, method = self.analyze_frame(frame)
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)

        color = (0, 255, 0)
        if text == "SPACE": color = (200, 200, 200)

        cv2.putText(frame, f"AI: {text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if "AI" in method:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for lm in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)

        return frame, text

    # --- AUDIO ---
    def generate_audio(self, text, output_path):
        if not text or not text.strip(): return None
        try:
            abs_path = os.path.abspath(output_path)
            if os.path.exists(abs_path): os.remove(abs_path)
            cmd = ["say", "-o", abs_path, "--data-format=LEI16@44100", text]
            subprocess.run(cmd, check=True)
            return abs_path
        except:
            return None

    def run_research_benchmark(self, input_path, asset_dir, noise=False):
        return {"frames": [0], "mse_time": [0], "mse_conf": [0], "ncc_time": [0], "ncc_conf": [0], "orb_time": [0],
                "orb_conf": [0]}