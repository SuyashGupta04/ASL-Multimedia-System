import cv2
import mediapipe as mp
import numpy as np
import os
import time
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
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize ORB Detector for Feature Matching
        self.orb = cv2.ORB_create(nfeatures=1000)

    # ==========================================
    # 1. CORE: FRAME PROCESSING (Live & Video)
    # ==========================================
    def process_frame(self, frame):
        """
        Analyzes a single frame for Sign Language.
        Returns: (Annotated Frame, Detected Text)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        height, width, _ = frame.shape
        detected_char = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw Skeleton
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Extract Key Landmarks
                landmarks = hand_landmarks.landmark

                # Logic: Simple Geometry for Basic Signs (A, B, C, Hello, Thanks)
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                middle_tip = landmarks[12]
                ring_tip = landmarks[16]
                pinky_tip = landmarks[20]
                wrist = landmarks[0]

                # Calculate distances (Normalized by frame size isn't strictly needed for relative logic,
                # but good for robust checks. Here we use simple relative positions).

                # HELLO (Open Palm, hand up)
                if (index_tip.y < wrist.y and middle_tip.y < wrist.y and
                        pinky_tip.y < wrist.y and abs(thumb_tip.x - pinky_tip.x) > 0.1):
                    detected_char = "HELLO"

                # YES (Fist, bobbing - simplified to fist for static)
                elif (index_tip.y > landmarks[6].y and middle_tip.y > landmarks[10].y and
                      pinky_tip.y > landmarks[18].y):
                    detected_char = "YES"

                # UP / POINTING
                elif (index_tip.y < middle_tip.y and index_tip.y < ring_tip.y):
                    detected_char = "UP"

                # PEACE / VICTORY (Index + Middle up)
                elif (index_tip.y < landmarks[6].y and middle_tip.y < landmarks[10].y and
                      ring_tip.y > landmarks[14].y):
                    detected_char = "PEACE"

                # C (Curved hand)
                elif (distance.euclidean((thumb_tip.x, thumb_tip.y), (index_tip.x, index_tip.y)) < 0.15 and
                      index_tip.y < wrist.y):
                    detected_char = "C"

                else:
                    detected_char = "..."

        return frame, detected_char

    # ==========================================
    # 2. VIDEO PROCESSING (Smart Mode)
    # ==========================================
    def process_video_smart(self, video_path):
        """
        Reads a video, detects signs frame-by-frame, and generates a summary.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Error: Cannot open video."

        # Output Video Settings
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use avc1 for browser playback
        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = int(cap.get(5))

        # Temp output path
        temp_out = os.path.join("assets", "temp", "processed_output.mp4")
        os.makedirs(os.path.dirname(temp_out), exist_ok=True)

        out = cv2.VideoWriter(temp_out, fourcc, fps, (width, height))

        detected_words = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret: break

            # Process every 5th frame to speed up (and reduce flicker)
            if frame_count % 5 == 0:
                frame, char = self.process_frame(frame)
                if char and char != "...":
                    detected_words.append(char)

            # Write annotated frame
            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()

        # Summarize Result (Most frequent word detected)
        if detected_words:
            final_text = max(set(detected_words), key=detected_words.count)
        else:
            final_text = "No distinct sign detected."

        return temp_out, final_text

    # ==========================================
    # 3. RESEARCH LAB BENCHMARK (Fixed)
    # ==========================================
    def run_research_benchmark(self, video_path, asset_dir, inject_noise=False):
        """
        Benchmarks MSE, NCC, and ORB algorithms.
        Returns dictionary with timing data.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None

        mse_times = []
        ncc_times = []
        orb_times = []

        # Load a dummy reference image for comparison (e.g., Letter 'A')
        # If no image exists, create a dummy black image
        ref_path = os.path.join(asset_dir, "a.jpg")
        if os.path.exists(ref_path):
            ref_img = cv2.imread(ref_path, 0)  # Grayscale
        else:
            ref_img = np.zeros((100, 100), dtype=np.uint8)

        # Resize ref_img to a standard size for fair comparison
        ref_img = cv2.resize(ref_img, (200, 200))

        frame_count = 0
        max_frames = 50  # Limit to 50 frames to keep it fast

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret: break

            # Convert to Grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (200, 200))  # Match Ref Size

            # Inject Noise if requested
            if inject_noise:
                noise = np.random.normal(0, 25, gray.shape).astype(np.uint8)
                gray = cv2.add(gray, noise)

            # --- 1. MSE (Mean Squared Error) ---
            start = time.perf_counter()  # High precision timer
            err = np.sum((gray.astype("float") - ref_img.astype("float")) ** 2)
            err /= float(gray.shape[0] * gray.shape[1])
            end = time.perf_counter()
            mse_times.append((end - start) * 1000)  # Convert to ms

            # --- 2. NCC (Normalized Cross-Correlation) ---
            start = time.perf_counter()
            res = cv2.matchTemplate(gray, ref_img, cv2.TM_CCOEFF_NORMED)
            end = time.perf_counter()
            ncc_times.append((end - start) * 1000)

            # --- 3. ORB (Feature Matching) ---
            start = time.perf_counter()
            kp1, des1 = self.orb.detectAndCompute(gray, None)
            kp2, des2 = self.orb.detectAndCompute(ref_img, None)
            # Create Matcher (just initialization, but part of the process)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            if des1 is not None and des2 is not None:
                matches = bf.match(des1, des2)
            end = time.perf_counter()
            orb_times.append((end - start) * 1000)

            frame_count += 1

        cap.release()

        # If video was too short or empty, prevent division by zero
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