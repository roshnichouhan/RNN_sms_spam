# 📩 RNN SMS Spam Classifier

An end-to-end **Deep Learning project for SMS spam detection** using a **Recurrent Neural Network (RNN)**. The system processes SMS text, converts it into numerical sequences, and classifies messages as **Spam** or **Not Spam**.

The project includes model training, text preprocessing, prediction, and an interactive **Streamlit web application** for real-time spam detection.

---

## 🚀 Project Overview

Spam messages are unwanted messages that may contain advertisements, fraudulent offers, phishing attempts, or suspicious links.

This project uses **Natural Language Processing (NLP)** and **Deep Learning** to automatically identify whether an SMS message is spam.

### Key Features

* Text preprocessing and tokenization
* Sequence padding for neural-network input
* RNN-based text classification
* Binary classification: **Spam / Not Spam**
* Saved trained model
* Saved tokenizer for consistent preprocessing
* Real-time prediction
* Interactive Streamlit interface
* Separate training and prediction modules

---

## 🧠 Machine Learning Approach

The project follows this pipeline:

```text
SMS Message
     ↓
Text Cleaning
     ↓
Tokenization
     ↓
Sequence Conversion
     ↓
Padding
     ↓
RNN Model
     ↓
Prediction Probability
     ↓
Spam / Not Spam
```

The RNN learns patterns and relationships between words in SMS messages and uses these patterns to classify new messages.

---

## 🛠️ Technologies Used

| Technology         | Purpose                                |
| ------------------ | -------------------------------------- |
| Python             | Core programming language              |
| TensorFlow / Keras | Deep Learning model                    |
| RNN                | Text classification                    |
| NLP                | SMS text processing                    |
| NumPy              | Numerical operations                   |
| Pandas             | Data processing                        |
| Scikit-learn       | Data preparation and evaluation        |
| Streamlit          | Interactive web application            |
| Pickle             | Tokenizer and training-history storage |
| Git & GitHub       | Version control                        |

---

## 📂 Project Structure

```text
RNN_sms_spam/
│
├── data/
│   └── SMSSpamCollection.txt
│
├── models/
│   ├── spam_rnn.keras
│   ├── tokenizer.pkl
│   └── history.pkl
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

The project uses the **SMS Spam Collection dataset**, containing labeled SMS messages categorized as:

* `spam`
* `ham` (not spam)

The dataset is used to train the model to distinguish unwanted messages from legitimate messages.

---

## 🔄 Data Processing

The SMS text goes through several preprocessing steps before being provided to the neural network.

### 1. Text Tokenization

Words are converted into numerical tokens using a tokenizer.

Example:

```text
"Congratulations! You won a prize"
```

becomes a numerical sequence such as:

```text
[125, 8, 42, 17, 91]
```

### 2. Sequence Padding

SMS messages have different lengths. Padding converts them into sequences of a consistent length so they can be processed by the neural network.

---

## 🧬 RNN Model

The project uses a **Recurrent Neural Network** designed for sequential text data.

A typical flow is:

```text
Input Text
    ↓
Embedding
    ↓
RNN Layer
    ↓
Dense Layer
    ↓
Sigmoid Output
    ↓
Spam Probability
```

The final output represents the probability that a message belongs to the spam class.

---

## 💻 Streamlit Application

The project includes an interactive Streamlit application where users can enter an SMS message and receive an immediate prediction.

Example:

```text
Input:
Congratulations! You have won a free lottery prize. Call now!
```

Output:

```text
Prediction: SPAM
```

For a normal message:

```text
Input:
Hey, are you coming to class today?
```

Output:

```text
Prediction: NOT SPAM
```

---

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/roshnichouhan/RNN_sms_spam.git
```

```bash
cd RNN_sms_spam
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

```bash
python src/train.py
```

This trains the RNN model and saves the trained artifacts inside the `models/` directory.

### 5. Run the Prediction Script

```bash
python src/predict.py
```

### 6. Launch the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📈 Model Artifacts

The trained model and preprocessing objects are stored in the `models/` directory.

```text
models/
├── spam_rnn.keras
├── tokenizer.pkl
└── history.pkl
```

### `spam_rnn.keras`

Contains the trained RNN model.

### `tokenizer.pkl`

Stores the tokenizer used during training so that new SMS messages are processed consistently.

### `history.pkl`

Stores training-history information that can be used for analyzing model performance across epochs.

---

## 🔮 Example Prediction

```text
SMS:
"Congratulations! You have won a free lottery prize. Call now!"

Prediction:
SPAM
```

```text
SMS:
"Can you send me the assignment when you get home?"

Prediction:
NOT SPAM
```

---

## 📌 Skills Demonstrated

This project demonstrates practical experience with:

* Natural Language Processing
* Text preprocessing
* Tokenization
* Sequence padding
* Recurrent Neural Networks
* Binary classification
* TensorFlow / Keras
* Model serialization
* Prediction pipelines
* Streamlit deployment
* Git and GitHub

---

## 🔮 Future Improvements

Potential improvements include:

* Compare RNN with **LSTM and GRU**
* Add confusion matrix and classification metrics
* Display training/validation accuracy and loss graphs
* Improve text preprocessing
* Hyperparameter tuning
* Handle class imbalance
* Add confidence scores to predictions
* Deploy the Streamlit application online
* Experiment with pretrained NLP models

---

## 👩‍💻 Author

**Roshni Chauhan**

Aspiring **Data Science & Machine Learning professional** interested in building practical machine-learning and deep-learning applications.

### GitHub

[github.com/roshnichouhan](https://github.com/roshnichouhan)

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.


https://github.com/user-attachments/assets/2c7c8a85-8ca2-4dfd-8da3-f7a09587b4a3

