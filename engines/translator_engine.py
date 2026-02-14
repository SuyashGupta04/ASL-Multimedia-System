import cv2
import mediapipe as mp
import numpy as np
import os
import time
import math
import pickle
from scipy.spatial import distance
from collections import deque
from moviepy.editor import VideoFileClip, AudioFileClip
from gtts import gTTS

# NEW: Import OCR library
try:
    import pytesseract
except ImportError:
    pytesseract = None


class TranslatorEngine:
    def __init__(self):
        # 1. Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # 2. Initialize Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_draw = mp.solutions.drawing_utils
        self.orb = cv2.ORB_create(nfeatures=1000)

        # 3. State Management
        self.history = []
        self.history_len = 12

        # 4. Trajectory Buffers
        self.trajectory_wrist = deque(maxlen=20)
        self.trajectory_index = deque(maxlen=20)

        # 5. Load AI Model
        self.model = None
        self.model_path = "model.p"
        self.labels_dict = {0: 'A', 1: 'B', 2: 'C'}

        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
            except:
                pass

    # ... [Keep helper functions: draw_trajectory, detect_expression, get_finger_status, predict_with_ai] ...
    # (For brevity, I'm skipping the repetitive helper functions from previous steps.
    #  Ensure you keep them or copy the full block below.)

    def draw_trajectory(self, frame, trajectory, color):
        for i in range(1, len(trajectory)):
            if trajectory[i - 1] is None or trajectory[i] is None: continue
            thickness = int(np.sqrt(20 / float(len(trajectory) - i + 1)) * 2)
            cv2.line(frame, trajectory[i - 1], trajectory[i], color, thickness)
        return frame

    def detect_expression(self, face_landmarks):
        right_brow = face_landmarks.landmark[105].y
        right_eye = face_landmarks.landmark[159].y
        left_brow = face_landmarks.landmark[334].y
        left_eye = face_landmarks.landmark[386].y
        if (abs(right_brow - right_eye) + abs(left_brow - left_eye)) / 2 > 0.055: return "QUESTION"
        return "NEUTRAL"

    def get_finger_status(self, lm):
        fingers = []
        if lm[4].x > lm[3].x:
            fingers.append(True)
        else:
            fingers.append(False)
        fingers.append(lm[8].y < lm[6].y)
        fingers.append(lm[12].y < lm[10].y)
        fingers.append(lm[16].y < lm[14].y)
        fingers.append(lm[20].y < lm[18].y)
        return fingers

    def predict_with_ai(self, hand_landmarks):
        if not self.model: return None
        try:
            data_aux = []
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                data_aux.append(hand_landmarks.landmark[i].y - min(y_))
            prediction = self.model.predict([np.asarray(data_aux)])
            return self.labels_dict[int(prediction[0])]
        except:
            return None

    # ... [Keep Geometric Rules: detect_character_geometric, detect_word_geometric] ...
    def detect_character_geometric(self, fingers, lm):
        if fingers == [True, False, False, False, False] and lm[4].y < lm[3].y: return "A"
        if fingers == [False, True, True, True, True]: return "B"
        dist_c = math.hypot(lm[8].x - lm[4].x, lm[8].y - lm[4].y)
        if 0.03 < dist_c < 0.2 and fingers[1] and fingers[2]: return "C"
        if fingers == [False, True, False, False, False]: return "D"
        if fingers == [False, False, False, False, False]: return "E" if lm[4].y > lm[5].y else "S"
        dist_f = math.hypot(lm[8].x - lm[4].x, lm[8].y - lm[4].y)
        if dist_f < 0.05 and fingers[2] and fingers[3] and fingers[4]: return "F"
        if fingers == [False, False, False, False, True]: return "I"
        if fingers == [True, True, False, False, False]: return "L"
        dist_o = math.hypot(lm[12].x - lm[4].x, lm[12].y - lm[4].y)
        if dist_o < 0.05 and not fingers[1]: return "O"
        if fingers == [False, True, True, False, False]:
            return "U" if math.hypot(lm[8].x - lm[12].x, lm[8].y - lm[12].y) < 0.04 else "V"
        if fingers == [False, True, True, True, False]: return "W"
        if fingers == [True, False, False, False, True]: return "Y"
        return ""

    def detect_word_geometric(self, fingers, lm):
        if all(fingers) and lm[0].y < 0.4: return "HELLO"
        if fingers == [True, True, False, False, True]: return "I LOVE YOU"
        if fingers == [False, False, False, False, False]: return "YES"
        if fingers == [False, True, True, False, False]: return "PEACE"
        if math.hypot(lm[8].x - lm[4].x, lm[8].y - lm[4].y) < 0.05 and fingers[2] and fingers[3]: return "OKAY"
        if math.hypot(lm[12].x - lm[4].x, lm[12].y - lm[4].y) < 0.05 and not fingers[1]: return "NO"
        if fingers == [False, True, False, False, False]: return "UP"
        return ""

    # ... [Keep process_frame] ...
    def process_frame(self, frame, detection_mode="Letter", draw_trace=False):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_hand = self.hands.process(rgb_frame)
        results_face = self.face_mesh.process(rgb_frame)

        detected_text = ""
        expression = "NEUTRAL"
        confidence = 0.0
        conn_color = (0, 0, 255)
        lm_color = (0, 0, 255)

        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                expression = self.detect_expression(face_landmarks)

        if results_hand.multi_hand_landmarks:
            for hand_landmarks in results_hand.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                h, w, c = frame.shape
                self.trajectory_wrist.append((int(lm[0].x * w), int(lm[0].y * h)))
                self.trajectory_index.append((int(lm[8].x * w), int(lm[8].y * h)))

                ai_result = self.predict_with_ai(hand_landmarks)
                if ai_result:
                    raw_text = ai_result
                    confidence = 0.95
                    conn_color = (0, 215, 255)
                else:
                    fingers_state = self.get_finger_status(lm)
                    if detection_mode == "Letter":
                        raw_text = self.detect_character_geometric(fingers_state, lm)
                    else:
                        raw_text = self.detect_word_geometric(fingers_state, lm)

                if raw_text:
                    final_meaning = raw_text
                    if expression == "QUESTION":
                        if raw_text == "YES":
                            final_meaning = "REALLY?"
                        elif raw_text == "NO":
                            final_meaning = "NO?"
                        elif raw_text == "HELLO":
                            final_meaning = "HELLO?"
                    self.history.append(final_meaning)
                else:
                    self.history.append("...")

                if len(self.history) > self.history_len: self.history.pop(0)

                if self.history:
                    candidate = max(set(self.history), key=self.history.count)
                    count = self.history.count(candidate)
                    if not ai_result: confidence = count / self.history_len

                    if confidence > 0.6 and candidate != "...":
                        detected_text = candidate
                        if not ai_result: conn_color = (0, 255, 0)

                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=lm_color, thickness=2, circle_radius=4),
                    self.mp_draw.DrawingSpec(color=conn_color, thickness=2, circle_radius=2)
                )
        else:
            self.trajectory_wrist.clear()
            self.trajectory_index.clear()

        if draw_trace:
            frame = self.draw_trajectory(frame, self.trajectory_wrist, (255, 0, 0))
            frame = self.draw_trajectory(frame, self.trajectory_index, (0, 255, 255))

        return frame, detected_text, confidence

    # ==========================================
    # UPDATED VIDEO PROCESSING WITH OCR FALLBACK
    # ==========================================
    def process_video_smart(self, video_path):
        """
        1. Tries to detect Hand Signs frame-by-frame.
        2. If result is weak, scans the LAST 2 seconds for OCR text.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None, "Error: Cannot open video"

        # Settings
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = int(cap.get(5))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        temp_out = os.path.join("assets", "temp", "processed_output.mp4")
        os.makedirs(os.path.dirname(temp_out), exist_ok=True)
        out = cv2.VideoWriter(temp_out, fourcc, fps, (width, height))

        # Sign Language Variables
        detected_sequence = []
        current_stable_char = ""
        frame_stability_counter = 0
        STABILITY_THRESHOLD = 5

        # OCR Variables
        scan_for_ocr = True
        ocr_result = ""

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break

            # 1. Process Hands
            if frame_idx % 2 == 0:  # Optimize speed
                frame, char, conf = self.process_frame(frame, detection_mode="Letter", draw_trace=False)

                if char:
                    # Logic to build sentence
                    if char == current_stable_char:
                        frame_stability_counter += 1
                    else:
                        current_stable_char = char
                        frame_stability_counter = 0

                    if frame_stability_counter == STABILITY_THRESHOLD:
                        if not detected_sequence or char != detected_sequence[-1]:
                            detected_sequence.append(char)

            # 2. Process OCR (Only on the last 45 frames / 1.5 seconds)
            if scan_for_ocr and pytesseract and (total_frames - frame_idx < 45):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Threshold to isolate white text on black background
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

                try:
                    text = pytesseract.image_to_string(thresh).strip()
                    # Basic filter to ensure it's a valid word/sentence
                    if len(text) > 1 and len(text) > len(ocr_result):
                        ocr_result = text
                except:
                    pass

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        # 3. Decision Logic: Combine Hand Signs + OCR
        final_text = ""

        # Format Hand Signs
        if detected_sequence:
            final_text = " ".join(detected_sequence)

        # If Hand Signs are weak or empty, fallback to OCR
        if ocr_result:
            if not final_text:
                final_text = f"OCR: {ocr_result}"
            else:
                # Optionally show both if they differ
                final_text += f" (Text Slide: {ocr_result})"

        if not final_text:
            final_text = "No signs or text detected."

        return temp_out, final_text

    # ... [Keep run_research_benchmark, generate_audio] ...
    def run_research_benchmark(self, video_path, asset_dir, inject_noise=False):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None
        mse, ncc, orb = [], [], []
        ref = cv2.imread(os.path.join(asset_dir, "a.jpg"), 0)
        if ref is None: ref = np.zeros((200, 200), np.uint8)
        ref = cv2.resize(ref, (200, 200))

        for _ in range(50):
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (200, 200))
            if inject_noise: gray = cv2.add(gray, np.random.normal(0, 25, gray.shape).astype(np.uint8))
            s = time.perf_counter();
            np.sum((gray.astype("float") - ref.astype("float")) ** 2);
            mse.append((time.perf_counter() - s) * 1000)
            s = time.perf_counter();
            cv2.matchTemplate(gray, ref, cv2.TM_CCOEFF_NORMED);
            ncc.append((time.perf_counter() - s) * 1000)
            s = time.perf_counter();
            self.orb.detectAndCompute(gray, None);
            orb.append((time.perf_counter() - s) * 1000)
        cap.release()
        return {"mse_time": mse, "ncc_time": ncc, "orb_time": orb}

    def generate_audio(self, text, output_path):
        try:
            gTTS(text=text, lang='en').save(output_path)
        except:
            pass