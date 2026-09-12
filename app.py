
import pickle
import re

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = "models/spam_rnn.keras"
TOKENIZER_PATH = "models/tokenizer.pkl"
MAX_LENGTH = 100


# =========================================================
# LOAD RNN MODEL
# =========================================================

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Could not load RNN model: {e}")
        st.stop()


# =========================================================
# LOAD TOKENIZER
# =========================================================

@st.cache_resource
def load_tokenizer():
    try:
        with open(TOKENIZER_PATH, "rb") as file:
            tokenizer = pickle.load(file)

        return tokenizer

    except Exception as e:
        st.error(f"Could not load tokenizer: {e}")
        st.stop()


# Load model and tokenizer
model = load_model()
tokenizer = load_tokenizer()


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Remove special characters and numbers
    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="RNN SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)


# =========================================================
# TITLE
# =========================================================

st.title("📩 RNN SMS Spam Detector")

st.write(
    "A Deep Learning project using a Simple RNN "
    "to detect spam SMS messages."
)

st.divider()


# =========================================================
# SMS INPUT
# =========================================================

message = st.text_area(
    "Enter an SMS message:",
    height=150,
    placeholder="Example: Congratulations! You won a free prize!"
)


# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button("Predict", type="primary"):

    # Check empty input
    if not message.strip():

        st.warning("Please enter an SMS message.")

    else:

        # -------------------------------------------------
        # STEP 1: CLEAN TEXT
        # -------------------------------------------------

        cleaned = clean_text(message)


        # -------------------------------------------------
        # STEP 2: TOKENIZATION
        # -------------------------------------------------

        sequence = tokenizer.texts_to_sequences(
            [cleaned]
        )


        # -------------------------------------------------
        # STEP 3: PADDING
        # -------------------------------------------------

        padded = pad_sequences(
            sequence,
            maxlen=MAX_LENGTH,
            padding="post",
            truncating="post"
        )


        # -------------------------------------------------
        # STEP 4: RNN PREDICTION
        # -------------------------------------------------

        prediction = model.predict(
            padded,
            verbose=0
        )

        probability = float(prediction[0][0])


        # -------------------------------------------------
        # STEP 5: RESULT
        # -------------------------------------------------

        if probability >= 0.5:

            # Model predicts SPAM

            st.error("🚨 SPAM")

            st.metric(
                "Spam Probability",
                f"{probability:.2%}"
            )

            st.progress(
                probability
            )

        else:

            # Model predicts HAM / NOT SPAM

            ham_probability = 1 - probability

            st.success("✅ NOT SPAM")

            st.metric(
                "Not Spam Probability",
                f"{ham_probability:.2%}"
            )

            st.progress(
                ham_probability
            )


        # =================================================
        # RNN PROCESS DETAILS
        # =================================================

        with st.expander("🔍 See what the RNN received"):

            st.subheader("1. Original Message")

            st.write(message)


            st.subheader("2. Cleaned Text")

            st.code(cleaned)


            st.subheader("3. Tokenized Sequence")

            st.write(sequence[0])


            st.subheader("4. Padded Sequence")

            st.write(padded[0])


            st.subheader("5. RNN Output")

            st.write(
                f"Raw model output: {probability:.6f}"
            )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🧠 About the Model")

    st.write(
        """
        This application uses a Simple RNN
        for SMS spam classification.
        """
    )

    st.write("### Pipeline")

    st.write(
        """
        SMS
        ↓
        Text Cleaning
        ↓
        Tokenization
        ↓
        Padding
        ↓
        Embedding
        ↓
        Simple RNN
        ↓
        Dense Layer
        ↓
        Sigmoid
        ↓
        SPAM / NOT SPAM
        """
    )

    st.write("### Model Files")

    st.write(
        """
        spam_rnn.keras
        tokenizer.pkl
        """
    )

