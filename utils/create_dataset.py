import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DATA_DIR = './data_raw'
OUTPUT_FILE = 'data.pickle'

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data = []
labels = []

print(f"🔄 Processing images from {DATA_DIR}...")

for dir_ in os.listdir(DATA_DIR):
    if dir_.startswith('.'): continue  # Skip hidden files
    class_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_path): continue

    print(f"   -> Processing Class: '{dir_}'")

    for img_path in os.listdir(class_path):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(class_path, img_path))
        if img is None: continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 1. Collect all X, Y coords
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # 2. Normalize (Shift to Origin)
                min_x = min(x_)
                min_y = min(y_)

                # 3. Normalize (Scale to 0-1 range) - CRITICAL FIX
                max_val = max(max(x_) - min_x, max(y_) - min_y)
                if max_val == 0: max_val = 1  # Avoid division by zero

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    # Store normalized relative coordinates
                    data_aux.append((x - min_x) / max_val)
                    data_aux.append((y - min_y) / max_val)

            data.append(data_aux)
            labels.append(dir_)

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"✅ Success! Robust data saved to '{OUTPUT_FILE}'.")