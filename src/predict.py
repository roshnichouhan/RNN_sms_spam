import pickle
import re
import sys

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ==============================
# SETTINGS
# ==============================

MODEL_PATH = "models/spam_rnn.keras"
TOKENIZER_PATH = "models/tokenizer.pkl"
MAX_LENGTH = 100


# ==============================
# LOAD MODEL
# ==============================

model = tf.keras.models.load_model(MODEL_PATH)


# ==============================
# LOAD TOKENIZER
# ==============================

with open(TOKENIZER_PATH, "rb") as file:
    tokenizer = pickle.load(file)


# ==============================
# CLEAN TEXT
# ==============================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==============================
# PREDICT FUNCTION
# ==============================

def predict_sms(message):

    # 1. Clean message
    cleaned_message = clean_text(message)

    # 2. Convert words into numbers
    sequence = tokenizer.texts_to_sequences(
        [cleaned_message]
    )

    # 3. Padding
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    # 4. RNN prediction
    probability = model.predict(
        padded_sequence,
        verbose=0
    )[0][0]

    # 5. Classification
    if probability >= 0.5:
        label = "SPAM"
    else:
        label = "HAM / NOT SPAM"

    return label, probability


# ==============================
# MAIN PROGRAM
# ==============================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            'Usage: python src/predict.py "your SMS message"'
        )

        sys.exit()

    message = " ".join(sys.argv[1:])

    label, probability = predict_sms(message)

    print("\n==============================")
    print("       RNN SMS PREDICTION")
    print("==============================")

    print("\nMessage:")
    print(message)

    print("\nPrediction:")
    print(label)

    print(
        f"\nSpam probability: {probability:.2%}"
    )

    print("==============================")