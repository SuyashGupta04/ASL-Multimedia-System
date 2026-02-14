import os
import cv2
import time
import uuid

# --- CONFIGURATION ---
DATA_DIR = "data_raw"  # Folder to save images
IMG_SIZE = 128  # Resize images to 128x128 for consistency
CLASSES = []  # Will be filled by user input


def collect_data():
    # 1. Setup Directory
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 2. Input Label
    label_name = input("Enter the name of the sign you want to collect (e.g., 'A', 'Hello'): ").strip().upper()
    if not label_name:
        print("Error: Label cannot be empty.")
        return

    class_dir = os.path.join(DATA_DIR, label_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f"\n[INFO] preparing to collect data for: '{label_name}'")
    print(f"[INFO] stored in: {class_dir}")
    print("--------------------------------------------------")
    print("  PRESS 'S' to save a frame (snapshot)")
    print("  PRESS 'Q' to quit")
    print("--------------------------------------------------")

    cap = cv2.VideoCapture(0)
    counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Camera not accessible.")
            break

        # Display instructions on screen
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Collecting: {label_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Saved: {counter}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 'S' to Save | 'Q' to Quit", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('ASL Data Collector', display_frame)

        key = cv2.waitKey(1) & 0xFF

        # --- SAVE FRAME LOGIC ---
        if key == ord('s'):
            # Generate unique filename
            img_name = os.path.join(class_dir, f"{label_name}_{uuid.uuid1()}.jpg")

            # Optional: Crop or Process frame before saving?
            # For now, we save the raw frame.
            cv2.imwrite(img_name, frame)

            print(f"  -> Saved {img_name}")
            counter += 1
            time.sleep(0.1)  # Small delay to prevent double captures

        # --- QUIT LOGIC ---
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Collected {counter} images for '{label_name}'\n")


if __name__ == "__main__":
    while True:
        collect_data()
        cont = input("Do you want to collect another sign? (y/n): ")
        if cont.lower() != 'y':
            break