
import os
import re
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# =========================================================
# SETTINGS
# =========================================================

DATA_PATH = "data/SMSSpamCollection.txt"

MODEL_DIR = "models"

MAX_WORDS = 10000
MAX_LENGTH = 100
EMBEDDING_DIM = 64

EPOCHS = 15
BATCH_SIZE = 32


# =========================================================
# CREATE MODEL DIRECTORY
# =========================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

print("Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

print("Dataset shape:", df.shape)


# =========================================================
# REMOVE MISSING VALUES
# =========================================================

df = df.dropna()


# =========================================================
# CONVERT LABELS
# =========================================================

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


df["clean_message"] = df["message"].apply(clean_text)


# =========================================================
# INPUT AND TARGET
# =========================================================

X = df["clean_message"]
y = df["label"]


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# TOKENIZER
# =========================================================

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train)


# =========================================================
# TEXT → NUMBERS
# =========================================================

X_train_sequences = tokenizer.texts_to_sequences(X_train)

X_test_sequences = tokenizer.texts_to_sequences(X_test)


# =========================================================
# PADDING
# =========================================================

X_train_padded = pad_sequences(
    X_train_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_test_padded = pad_sequences(
    X_test_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)


print("Padded training shape:", X_train_padded.shape)


# =========================================================
# CLASS WEIGHTS
# =========================================================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(
    zip(classes, weights)
)

print("\nClass weights:")
print(class_weights)


# =========================================================
# BUILD SIMPLE RNN
# =========================================================

model = Sequential([

    Embedding(
        input_dim=MAX_WORDS,
        output_dim=EMBEDDING_DIM
    ),

    SimpleRNN(
        64,
        return_sequences=False
    ),

    Dropout(0.3),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(0.2),

    Dense(
        1,
        activation="sigmoid"
    )
])


# =========================================================
# COMPILE
# =========================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy"
    ]
)


# =========================================================
# MODEL SUMMARY
# =========================================================

model.summary()


# =========================================================
# EARLY STOPPING
# =========================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# =========================================================
# TRAIN
# =========================================================

print("\nTraining RNN...")

history = model.fit(
    X_train_padded,
    y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)


# =========================================================
# TEST
# =========================================================

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(
    X_test_padded,
    y_test,
    verbose=0
)

print("\nTest Loss:", test_loss)

print("Test Accuracy:", test_accuracy)


# =========================================================
# PREDICTIONS
# =========================================================

probabilities = model.predict(
    X_test_padded,
    verbose=0
).flatten()


predictions = (
    probabilities >= 0.5
).astype(int)


# =========================================================
# METRICS
# =========================================================

print("\nAccuracy:")

print(
    accuracy_score(
        y_test,
        predictions
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "HAM",
            "SPAM"
        ]
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# =========================================================
# SAVE MODEL
# =========================================================

model_path = os.path.join(
    MODEL_DIR,
    "spam_rnn.keras"
)

model.save(model_path)

print(
    "\nModel saved:",
    model_path
)


# =========================================================
# SAVE TOKENIZER
# =========================================================

tokenizer_path = os.path.join(
    MODEL_DIR,
    "tokenizer.pkl"
)

with open(
    tokenizer_path,
    "wb"
) as file:

    pickle.dump(
        tokenizer,
        file
    )

print(
    "Tokenizer saved:",
    tokenizer_path
)


# =========================================================
# SAVE HISTORY
# =========================================================

history_path = os.path.join(
    MODEL_DIR,
    "history.pkl"
)

with open(
    history_path,
    "wb"
) as file:

    pickle.dump(
        history.history,
        file
    )

print(
    "History saved:",
    history_path
)


print("\nTraining completed successfully!")

