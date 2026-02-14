import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
INPUT_FILE = 'data.pickle'
OUTPUT_MODEL = 'model.p'

print("🔄 Loading data...")
try:
    with open(INPUT_FILE, 'rb') as f:
        data_dict = pickle.load(f)
except FileNotFoundError:
    print(f"❌ Error: '{INPUT_FILE}' not found. Run create_dataset.py first.")
    exit()

# --- STEP 1: CLEAN THE DATA ---
print("🧹 Cleaning dataset (Removing inconsistent samples)...")

raw_data = data_dict['data']
raw_labels = data_dict['labels']

clean_data = []
clean_labels = []

# specific length for 1 hand (21 landmarks * 2 coords x/y)
EXPECTED_LENGTH = 42

for i, sample in enumerate(raw_data):
    if len(sample) == EXPECTED_LENGTH:
        clean_data.append(sample)
        clean_labels.append(raw_labels[i])
    else:
        # Optional: Print warning for bad samples
        # print(f"   ⚠️ Skipped sample {i} (Length: {len(sample)}, Expected: {EXPECTED_LENGTH})")
        pass

print(f"   Original samples: {len(raw_data)}")
print(f"   Clean samples:    {len(clean_data)}")

if len(clean_data) == 0:
    print("❌ Error: No valid data points found! Ensure your images contain exactly one hand.")
    exit()

# Convert to NumPy arrays
data = np.asarray(clean_data)
labels = np.asarray(clean_labels)

# --- STEP 2: SPLIT DATA ---
# Stratify ensures we have a mix of all labels (A, B, C) in both training and testing
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# --- STEP 3: TRAIN MODEL ---
print("🧠 Training Model (Random Forest)...")
model = RandomForestClassifier()
model.fit(x_train, y_train)

# --- STEP 4: EVALUATE ---
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)

print(f"✅ Accuracy: {score * 100:.2f}%")

# --- STEP 5: SAVE MODEL ---
with open(OUTPUT_MODEL, 'wb') as f:
    pickle.dump({'model': model}, f)

print(f"💾 Model saved successfully to '{OUTPUT_MODEL}'")
print("👉 Next Step: Move 'model.p' to your main project folder where 'app.py' is located.")