import json
import os
import hashlib
import streamlit as st

USER_DB_FILE = "users.json"


# --- 1. LOAD USERS ---
def load_users():
    if not os.path.exists(USER_DB_FILE):
        # Create default admin if file doesn't exist
        # Default Admin -> User: admin, Pass: admin123
        default_db = {
            "admin": {
                "password": hash_password("admin123"),
                "role": "admin",
                "name": "System Administrator"
            }
        }
        save_users(default_db)
        return default_db

    with open(USER_DB_FILE, "r") as f:
        return json.load(f)


# --- 2. SAVE USERS ---
def save_users(users_data):
    with open(USER_DB_FILE, "w") as f:
        json.dump(users_data, f, indent=4)


# --- 3. HASH PASSWORD (SECURITY) ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# --- 4. AUTHENTICATE USER ---
def login_user(username, password):
    users = load_users()
    if username in users:
        if users[username]["password"] == hash_password(password):
            return users[username]  # Return user dict (role, name)
    return None


# --- 5. REGISTER USER ---
def register_user(username, password, name):
    users = load_users()
    if username in users:
        return False, "Username already exists."

    users[username] = {
        "password": hash_password(password),
        "role": "user",  # Default role is always 'user'
        "name": name
    }
    save_users(users)
    return True, "Account created successfully! Please login."