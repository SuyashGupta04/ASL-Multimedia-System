import json
import os
from datetime import datetime
import pandas as pd

FEEDBACK_FILE = "feedback.json"


def init_feedback_db():
    """Creates the JSON file if it doesn't exist."""
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "w") as f:
            json.dump([], f)


def save_feedback(username, rating, text):
    """Saves a new feedback entry."""
    init_feedback_db()

    with open(FEEDBACK_FILE, "r") as f:
        try:
            data = json.load(f)
        except:
            data = []

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": username,
        "rating": rating,
        "comment": text
    }
    data.append(entry)

    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=4)
    return True


def get_feedback_df():
    """Returns feedback as a Pandas DataFrame for the Admin Dashboard."""
    init_feedback_db()
    try:
        df = pd.read_json(FEEDBACK_FILE)
        # If empty, return structured empty df
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "user", "rating", "comment"])
        # Sort by latest first
        return df.iloc[::-1]
    except:
        return pd.DataFrame(columns=["timestamp", "user", "rating", "comment"])