import pandas as pd
import os
from datetime import datetime

FEEDBACK_FILE = "assets/feedback_data.csv"


def init_feedback_db():
    if not os.path.exists("assets"):
        os.makedirs("assets")

    if not os.path.exists(FEEDBACK_FILE):
        df = pd.DataFrame(columns=["timestamp", "user", "rating", "category", "sus_score", "comment"])
        df.to_csv(FEEDBACK_FILE, index=False)


def save_feedback(user, rating, category, sus_score, comment):
    init_feedback_db()
    df = pd.read_csv(FEEDBACK_FILE)

    new_data = pd.DataFrame({
        "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "user": [user],
        "rating": [rating],
        "category": [category],
        "sus_score": [sus_score],  # System Usability Score
        "comment": [comment]
    })

    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(FEEDBACK_FILE, index=False)


def get_feedback_df():
    init_feedback_db()
    return pd.read_csv(FEEDBACK_FILE)