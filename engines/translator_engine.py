import cv2
import mediapipe as mp
import numpy as np
import os
import time
import math
from scipy.spatial import distance
from moviepy.editor import VideoFileClip, AudioFileClip
from gtts import gTTS


class TranslatorEngine:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize ORB Detector for Feature Matching (Research Lab)
        self.orb = cv2.ORB_create(nfeatures=1000)

        # History for smoothing predictions and calculating confidence
        self.history = []
        self.history_len = 12

    # ==========================================
    # HELPER: FINGER STATUS
    # ==========================================
    def get_finger_status(self, lm):
        """
        Returns a list of 5 booleans [Thumb, Index, Middle, Ring, Pinky]
        True = Open/Extended, False = Closed/Bent
        """
        fingers = []
        # Thumb (Check x-axis for right hand assumption, logic flips for left)
        if lm[4].x > lm[3].x:
            fingers.append(True)
        else:
            fingers.append(False)

        # Other 4 fingers (Check Y-axis: Tip above PIP joint)
        fingers.append(lm[8].y < lm[6].y)  # Index
        fingers.append(lm[12].y < lm[10].y)  # Middle
        fingers.append(lm[16].y < lm[14].y)  # Ring
        fingers.append(lm[20].y < lm[18].y)  # Pinky
        return fingers

    # ==========================================
    # LOGIC: ALPHABETS (A-Z)
    # ==========================================
    def detect_character(self, fingers, lm):
        if fingers == [True, False, False, False, False]: return "A"
        if fingers == [False, True, True, True, True]: return "B"

        # C: Curved hand
        dist_c = math.hypot(lm[8].x - lm[4].x, lm[8].y - lm[4].y)
        if 0.03 < dist_c < 0.15 and fingers[1] and fingers[2]: return "C"

        if fingers == [False, True, False, False, False]: return "D"
        if fingers == [False, False, False, False, False]: return "E"

        # F: Index + Thumb touching
        dist_f = math.hypot(lm[8].x - lm[4].x, lm[8].y - lm[4].y)
        if dist_f < 0.05 and fingers[2] and fingers[3] and fingers[4]: return "F"

        if fingers == [False, False, False, False, True]: return "I"
        if fingers == [True, True, False, False, False]: return "L"

        # O: Tips touching thumb
        dist_o = math.hypot(lm[12].x - lm[4].x, lm[12].y - lm[4].y)
        if dist_o < 0.05 and not fingers[1]: return "O"

        if fingers == [False, True, True, False, False]: return "V"
        if fingers == [False, True, True, True, False]: return "W"
        if fingers == [True, False, False, False, True]: return "Y"

        return ""

    # ==========================================
    # LOGIC: WHOLE WORDS
    # ==========================================
    def detect_word(self, fingers, lm):
        if all(fingers) and lm[0].y < 0.4: return "HELLO"
        if fingers == [True, True, False, False, True]: return "I LOVE YOU"
        if fingers == [False, False, False, False, False]: return "YES"
        if fingers == [False, True, True, False, False]: return "PEACE"

        dist_f = math.hypot(lm[8].x - lm[4].x, lm[8].y - lm[4].y)
        if dist_f < 0.05 and fingers[2] and fingers[3]: return "OKAY"

        dist_no = math.hypot(lm[12].x - lm[4].x, lm[12].y - lm[4].y)
        if dist_no < 0.05 and fingers[1] == False: return "NO"

        if fingers == [False, True, False, False, False]: return "UP"

        return ""

    # ==========================================
    # 1. CORE: FRAME PROCESSING (Advanced)
    # ==========================================
    def process_frame(self, frame, detection_mode="Letter"):
        """
        Analyzes a single frame.
        Returns: (Annotated Frame, Detected Text, Confidence Score)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        detected_text = ""
        confidence = 0.0

        # Default Visuals: Red (Searching)
        conn_color = (0, 0, 255)
        lm_color = (0, 0, 255)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                fingers_state = self.get_finger_status(lm)

                # Route Logic
                if detection_mode == "Letter":
                    raw_text = self.detect_character(fingers_state, lm)
                else:
                    raw_text = self.detect_word(fingers_state, lm)

                # Stabilizer & Confidence Calculation
                if raw_text:
                    self.history.append(raw_text)
                else:
                    self.history.append("...")  # Append silence

                if len(self.history) > self.history_len:
                    self.history.pop(0)

                # Determine Result based on History
                if self.history:
                    candidate = max(set(self.history), key=self.history.count)
                    count = self.history.count(candidate)

                    # Confidence = Percentage of frames matching the candidate
                    confidence = count / self.history_len

                    # Threshold: Return text only if confidence > 60%
                    if confidence > 0.6 and candidate != "...":
                        detected_text = candidate
                        # Success Visuals: Green (Found)
                        conn_color = (0, 255, 0)
                        lm_color = (0, 255, 0)

                # Draw Skeleton with Dynamic Colors
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=lm_color, thickness=2, circle_radius=4),
                    self.mp_draw.DrawingSpec(color=conn_color, thickness=2, circle_radius=2)
                )

        return frame, detected_text, confidence

    # ==========================================
    # 2. VIDEO PROCESSING (Smart Mode)
    # ==========================================
    def process_video_smart(self, video_path):
        """
        Reads a video, detects signs frame-by-frame.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None, "Error"

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = int(cap.get(5))

        temp_out = os.path.join("assets", "temp", "processed_output.mp4")
        os.makedirs(os.path.dirname(temp_out), exist_ok=True)
        out = cv2.VideoWriter(temp_out, fourcc, fps, (width, height))

        detected_words = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret: break

            if frame_count % 5 == 0:
                # Unpack all 3 values (frame, text, confidence)
                frame, char, conf = self.process_frame(frame, detection_mode="Letter")
                if char: detected_words.append(char)

            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()

        if detected_words:
            final_text = max(set(detected_words), key=detected_words.count)
        else:
            final_text = "No distinct sign detected."

        return temp_out, final_text

    # ==========================================
    # 3. RESEARCH LAB BENCHMARK
    # ==========================================
    def run_research_benchmark(self, video_path, asset_dir, inject_noise=False):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None

        mse_times = []
        ncc_times = []
        orb_times = []

        # Load Dummy Reference
        ref_path = os.path.join(asset_dir, "a.jpg")
        if os.path.exists(ref_path):
            ref_img = cv2.imread(ref_path, 0)
        else:
            ref_img = np.zeros((100, 100), dtype=np.uint8)
        ref_img = cv2.resize(ref_img, (200, 200))

        frame_count = 0
        max_frames = 50

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret: break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (200, 200))

            if inject_noise:
                noise = np.random.normal(0, 25, gray.shape).astype(np.uint8)
                gray = cv2.add(gray, noise)

            # 1. MSE
            start = time.perf_counter()
            err = np.sum((gray.astype("float") - ref_img.astype("float")) ** 2)
            err /= float(gray.shape[0] * gray.shape[1])
            end = time.perf_counter()
            mse_times.append((end - start) * 1000)

            # 2. NCC
            start = time.perf_counter()
            res = cv2.matchTemplate(gray, ref_img, cv2.TM_CCOEFF_NORMED)
            end = time.perf_counter()
            ncc_times.append((end - start) * 1000)

            # 3. ORB
            start = time.perf_counter()
            kp1, des1 = self.orb.detectAndCompute(gray, None)
            kp2, des2 = self.orb.detectAndCompute(ref_img, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            if des1 is not None and des2 is not None:
                matches = bf.match(des1, des2)
            end = time.perf_counter()
            orb_times.append((end - start) * 1000)

            frame_count += 1

        cap.release()
        if not mse_times: return None

        return {
            "mse_time": mse_times,
            "ncc_time": ncc_times,
            "orb_time": orb_times
        }

    # ==========================================
    # 4. AUDIO GENERATION
    # ==========================================
    def generate_audio(self, text, output_path):
        if not text: return
        try:
            tts = gTTS(text=text, lang='en')
            tts.save(output_path)
        except Exception as e:
            print(f"Audio Error: {e}")