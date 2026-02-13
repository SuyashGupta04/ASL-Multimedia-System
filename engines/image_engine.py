import os


class ImageEngine:
    def __init__(self):
        # Path to your assets folder
        self.asset_dir = os.path.join("assets", "images")

    def get_asset(self, text):
        """
        Smart Asset Lookup:
        1. Checks for Video (.mp4) -> Returns path if found.
        2. Checks for Image (.jpg, .png) -> Returns path if found.
        3. Returns None if nothing is found.
        """
        key = text.lower().strip()

        # Priority 1: MP4 Video (Motion Sign)
        vid_path = os.path.join(self.asset_dir, f"{key}.mp4")
        if os.path.exists(vid_path):
            return vid_path

        # Priority 2: Static Image (Fallback)
        for ext in [".jpg", ".jpeg", ".png"]:
            img_path = os.path.join(self.asset_dir, f"{key}{ext}")
            if os.path.exists(img_path):
                return img_path

        return None