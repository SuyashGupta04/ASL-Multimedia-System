import os
from PIL import Image, ImageDraw

# --- SETUP FOLDERS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "assets", "images")
VID_DIR = os.path.join(BASE_DIR, "assets", "videos")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(VID_DIR, exist_ok=True)

# --- 1. GENERATE A-Z IMAGES ---
print("Generating dummy images...")
chars = "abcdefghijklmnopqrstuvwxyz0123456789"
for char in chars:
    img = Image.new('RGB', (200, 200), color=(50, 50, 50))  # Grey background
    d = ImageDraw.Draw(img)
    # Draw letter in middle (placeholder for real sign)
    d.text((90, 90), char.upper(), fill=(255, 255, 255))
    img.save(os.path.join(IMG_DIR, f"{char}.png"))
print("✅ Images created in assets/images/")

# --- 2. CREATE DUMMY VIDEOS ---
# NOTE: In a real app, these would be real human videos.
# For this prototype, we will create simple short video files from images.
try:
    from moviepy.editor import ImageClip

    print("\nGenerating dummy videos (this might take a minute)...")

    # Create a dummy video for "HELLO"
    hello_img = Image.new('RGB', (640, 480), color=(0, 100, 0))  # Green
    d = ImageDraw.Draw(hello_img)
    d.text((300, 240), "ASL SIGN: HELLO", fill=(255, 255, 255))
    hello_img_path = "temp_hello.png"
    hello_img.save(hello_img_path)

    clip = ImageClip(hello_img_path).set_duration(2).set_fps(24)
    clip.write_videofile(os.path.join(VID_DIR, "HELLO.mp4"), verbose=False, logger=None)

    # Create a dummy video for "WORLD"
    world_img = Image.new('RGB', (640, 480), color=(0, 0, 100))  # Blue
    d = ImageDraw.Draw(world_img)
    d.text((300, 240), "ASL SIGN: WORLD", fill=(255, 255, 255))
    world_img_path = "temp_world.png"
    world_img.save(world_img_path)

    clip = ImageClip(world_img_path).set_duration(2).set_fps(24)
    clip.write_videofile(os.path.join(VID_DIR, "WORLD.mp4"), verbose=False, logger=None)

    # Cleanup temp images
    os.remove(hello_img_path)
    os.remove(world_img_path)
    print("✅ Dummy videos created: HELLO.mp4 and WORLD.mp4")

except ImportError:
    print("\n⚠️ Could not generate dummy videos. Is moviepy installed?")
    print("Please manually put 'HELLO.mp4' and 'WORLD.mp4' into assets/videos/")
except Exception as e:
    print(f"\n⚠️ An error occurred generating videos: {e}")