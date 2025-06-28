import os
import re
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from flask import Flask, request, render_template
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import pickle
import json

# ===== Hyperparameters =====
MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 64
LSTM_UNITS = 128
BATCH_SIZE = 128
EPOCHS = 20
MODEL_FILENAME = "sentiment_rnn_model.h5"
TOKENIZER_FILENAME = "tokenizer.pkl"
COMMENTS_FILENAME = "comments2.csv"

# ===== Data Cleaning =====
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

# ===== Load and Prepare Data =====
def load_and_prepare_data(filename):
    df = pd.read_csv(filename, encoding="latin1")
    if "Comment" not in df.columns or "Sentiment" not in df.columns:
        raise ValueError("CSV must contain 'Comment' and 'Sentiment' columns.")
    df = df.dropna(subset=["Comment", "Sentiment"])
    print("Sentiment distribution:")
    print(df["Sentiment"].value_counts())
    texts = df["Comment"].astype(str).apply(clean_text).tolist()
    labels = df["Sentiment"].astype(int).tolist()
    return texts, np.array(labels)

# ===== Build Model =====
def build_model(vocab_size, max_sequence_length):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_sequence_length),
        LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ===== Training Function =====
def train_model():
    print("Starting training process...")
    texts, labels = load_and_prepare_data(COMMENTS_FILENAME)
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post")

    X_train, X_val, y_train, y_val = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)
    class_weights_vals = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: class_weights_vals[i] for i in range(len(class_weights_vals))}
    print("Using class weights:", class_weights)

    vocab_size = min(MAX_NUM_WORDS, len(tokenizer.word_index) + 1)
    model = build_model(vocab_size, MAX_SEQUENCE_LENGTH)
    early_stop = EarlyStopping(monitor='loss', patience=3)

    target_accuracy = 0.90
    current_accuracy = 0
    previous_accuracy = -1
    max_total_epochs = 50
    epochs_per_round = 2
    total_epochs_trained = 0
    breaker = False

    while current_accuracy < target_accuracy and total_epochs_trained < max_total_epochs and not breaker:
        print(f"\n🔁 Training... Current Accuracy: {current_accuracy:.2f}, Target: {target_accuracy:.2f}")
        model.fit(X_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=epochs_per_round,
                  validation_data=(X_val, y_val),
                  class_weight=class_weights,
                  verbose=1)

        loss, acc = model.evaluate(X_val, y_val, verbose=0)
        current_accuracy = acc
        if abs(previous_accuracy - current_accuracy) <= 0.01:
            breaker = True
        previous_accuracy = acc
        total_epochs_trained += epochs_per_round
        print(f"✅ Validation Accuracy after {total_epochs_trained} epochs: {current_accuracy:.4f}")

    model.save(MODEL_FILENAME)
    with open(TOKENIZER_FILENAME, "wb") as f:
        pickle.dump(tokenizer, f)
    print("✅ Model and tokenizer saved.")

    return model, tokenizer
# ===== Bar Chart Maker ======


def save_confidence_plot(probabilities, output_path='static/confidence.png'):
    labels = ['Negative', 'Neutral', 'Positive']
    plt.figure(figsize=(6,4))
    plt.bar(labels, probabilities, color=['red', 'gray', 'green'])
    plt.title("Model Confidence")
    plt.ylim(0, 1)
    for i, v in enumerate(probabilities):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()




# ===== Flask App =====
app = Flask(__name__)
model = None
tokenizer = None

@app.route("/", methods=["GET"])
def index():
    return render_template("app.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    
    user_comment = request.form.get("comment", "")
    if not user_comment:
        return render_template("app.html", prediction="No comment provided.")

    cleaned = clean_text(user_comment)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded_seq = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding="post")

    pred_probs = model.predict(padded_seq, verbose=0)[0]
    sentiment = int(np.argmax(pred_probs))
    print("Prediction probabilities:", pred_probs)
    save_confidence_plot(pred_probs)
    if sentiment == 0:
        return render_template("app.html", prediction="Negative", confidence = pred_probs[0]*100)
    elif sentiment == 1:
        return render_template("app.html", prediction="Neutral",confidence = (pred_probs[1]*100))
    elif sentiment == 2:
        return render_template("app.html", prediction="Positive",confidence = (pred_probs[2]*100))
    else:
        return render_template("app.html", prediction="Unexpected prediction result.",confidence = (pred_probs[0]*100))

# ===== Main Entry Point =====
if __name__ == '__main__':
    confidence_path = os.path.join("static", "confidence.png")
    if os.path.exists(confidence_path):
        try:
            os.remove(confidence_path)
            print("🧹 Removed old confidence.png")
        except Exception as e:
            print(f"⚠️ Failed to remove old confidence.png: {e}")
    if '-train' in sys.argv:
        model, tokenizer = train_model()
    elif "-evaluate" in sys.argv:
        print("🔍 Evaluating model for visualization...")
        model = tf.keras.models.load_model(MODEL_FILENAME)
        with open(TOKENIZER_FILENAME, "rb") as f:
            tokenizer = pickle.load(f)

        texts, labels = load_and_prepare_data(COMMENTS_FILENAME)
        sequences = tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post")

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        loss, acc = model.evaluate(padded_sequences, np.array(labels), verbose=1)
        print(f"✅ Model evaluation -- Loss: {loss:.4f}, Accuracy: {acc:.4f}")

        # Get predicted classes
        predictions = model.predict(padded_sequences)
        predicted_labels = np.argmax(predictions, axis=1)

        # Save to a JSON file for make_visuals.py
        with open("eval_output.json", "w") as f:
            json.dump({
                "true_labels": labels[:1000].tolist(),            # ✅ FIXED
                "predicted_labels": predicted_labels[:1000].tolist()
            }, f)


        print("📁 Saved evaluation results to eval_output.json.")
        exit(0)
    else:
        if not os.path.exists(MODEL_FILENAME) or not os.path.exists(TOKENIZER_FILENAME):
            print("❌ Model or tokenizer file not found. Use -train to train a new model.")
            sys.exit(1)
        model = tf.keras.models.load_model(MODEL_FILENAME)
        with open(TOKENIZER_FILENAME, "rb") as f:
            tokenizer = pickle.load(f)
        print("✅ Model and tokenizer loaded.")

    app.run(debug=False, port=5000)
