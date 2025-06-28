import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os

static_files = ["confusion_matrix.png", "distribution.png"]

for filename in static_files:
    path = os.path.join("static", filename)
    if os.path.exists(path):
        try:
            os.remove(path)
            print(f"🧹 Removed old static/{filename}")
        except Exception as e:
            print(f"⚠️ Failed to remove static/{filename}: {e}")

# Load the evaluation output
with open("eval_output.json", "r") as f:
    data = json.load(f)

true_labels = np.array(data["true_labels"])
predicted_labels = np.array(data["predicted_labels"])

# --- Visualization 1: Confusion Matrix ---
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("static/confusion_matrix.png")
print("📊 Saved confusion_matrix.png")

# --- Visualization 2: Class distribution ---
plt.figure()
labels, counts = np.unique(predicted_labels, return_counts=True)
plt.bar(['Negative', 'Neutral', 'Positive'], counts, color=['red', 'gray', 'green'])
plt.title("Predicted Sentiment Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("static/sentiment_distribution.png")
print("📊 Saved sentiment_distribution.png")

# --- Visualization 3: Classification Report as text (optional) ---
print("\n📋 Classification Report:\n")
print(classification_report(true_labels, predicted_labels, target_names=["Negative", "Neutral", "Positive"]))
