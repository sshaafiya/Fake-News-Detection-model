# ============================================================
# FAKE NEWS DETECTION USING TENSORFLOW + LSTM
# ============================================================

# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import re
import string
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping


# ------------------------------------------------------------
# 2. SETTINGS
# ------------------------------------------------------------

DATASET_PATH = "dataset/fake_or_real_news.csv"

MAX_WORDS = 10000
MAX_LENGTH = 200
EMBEDDING_DIM = 64

TEST_SIZE = 0.20
RANDOM_STATE = 42

EPOCHS = 10
BATCH_SIZE = 32


# ------------------------------------------------------------
# 3. LOAD DATASET
# ------------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully!")

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ------------------------------------------------------------
# 4. CHECK REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "title",
    "text",
    "subject",
    "date",
    "label"
]

for column in required_columns:

    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' was not found in the dataset."
        )


# ------------------------------------------------------------
# 5. CHECK LABELS
# ------------------------------------------------------------

print("\nLabel distribution:")

print(df["label"].value_counts())


# ------------------------------------------------------------
# 6. HANDLE MISSING VALUES
# ------------------------------------------------------------

df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")
df["subject"] = df["subject"].fillna("")
df["date"] = df["date"].fillna("")
df["label"] = df["label"].fillna("")


# ------------------------------------------------------------
# 7. COMBINE TEXT COLUMNS
# ------------------------------------------------------------

df["content"] = (
    df["title"].astype(str)
    + " "
    + df["text"].astype(str)
    + " "
    + df["subject"].astype(str)
)


# ------------------------------------------------------------
# 8. TEXT CLEANING FUNCTION
# ------------------------------------------------------------

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


print("\nCleaning text...")

df["content"] = df["content"].apply(clean_text)


# ------------------------------------------------------------
# 9. CONVERT LABELS
# ------------------------------------------------------------

def convert_label(label):

    label = str(label).strip().upper()

    if label == "FAKE":
        return 0

    elif label == "REAL":
        return 1

    else:
        return np.nan


df["label_encoded"] = df["label"].apply(convert_label)


# ------------------------------------------------------------
# 10. REMOVE INVALID LABELS
# ------------------------------------------------------------

df = df.dropna(
    subset=["content", "label_encoded"]
)

df["label_encoded"] = df["label_encoded"].astype(int)


# ------------------------------------------------------------
# 11. DISPLAY CLEANED DATA
# ------------------------------------------------------------

print("\nCleaned dataset:")

print(
    df[
        ["content", "label", "label_encoded"]
    ].head()
)


# ------------------------------------------------------------
# 12. GET INPUT AND OUTPUT
# ------------------------------------------------------------

X = df["content"].values

y = df["label_encoded"].values


# ------------------------------------------------------------
# 13. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


print("\nData split completed!")

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ------------------------------------------------------------
# 14. TOKENIZER
# ------------------------------------------------------------

print("\nCreating tokenizer...")

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train)


# ------------------------------------------------------------
# 15. CONVERT TEXT TO SEQUENCES
# ------------------------------------------------------------

X_train_sequences = tokenizer.texts_to_sequences(
    X_train
)

X_test_sequences = tokenizer.texts_to_sequences(
    X_test
)


# ------------------------------------------------------------
# 16. PAD SEQUENCES
# ------------------------------------------------------------

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


print("\nTraining input shape:")
print(X_train_padded.shape)

print("\nTesting input shape:")
print(X_test_padded.shape)


# ------------------------------------------------------------
# 17. BUILD LSTM MODEL
# ------------------------------------------------------------

print("\nBuilding LSTM model...")


model = Sequential([

    # Word Embedding
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_LENGTH
    ),

    # LSTM
    LSTM(
        64
    ),

    # Dropout
    Dropout(
        0.5
    ),

    # Dense Layer
    Dense(
        32,
        activation="relu"
    ),

    # Output Layer
    Dense(
        1,
        activation="sigmoid"
    )
])


# ------------------------------------------------------------
# 18. COMPILE MODEL
# ------------------------------------------------------------

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ------------------------------------------------------------
# 19. DISPLAY MODEL
# ------------------------------------------------------------

print("\nModel Summary:")

model.summary()


# ------------------------------------------------------------
# 20. EARLY STOPPING
# ------------------------------------------------------------

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=2,

    restore_best_weights=True
)


# ------------------------------------------------------------
# 21. TRAIN MODEL
# ------------------------------------------------------------

print("\nStarting training...")

history = model.fit(

    X_train_padded,

    y_train,

    validation_split=0.2,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# ------------------------------------------------------------
# 22. EVALUATE MODEL
# ------------------------------------------------------------

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(
    X_test_padded,
    y_test,
    verbose=0
)

print(
    f"\nTest Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy:.4f}"
)


# ------------------------------------------------------------
# 23. MAKE PREDICTIONS
# ------------------------------------------------------------

print("\nGenerating predictions...")

y_probability = model.predict(
    X_test_padded,
    verbose=0
)


# Convert probability to class

y_pred = (
    y_probability >= 0.5
).astype(int).flatten()


# ------------------------------------------------------------
# 24. CALCULATE METRICS
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# ------------------------------------------------------------
# 25. PRINT RESULTS
# ------------------------------------------------------------

print("\n")
print("=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1-Score  : {f1:.4f}"
)

print("=" * 50)


# ------------------------------------------------------------
# 26. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "FAKE",
            "REAL"
        ],
        zero_division=0
    )
)


# ------------------------------------------------------------
# 27. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\nConfusion Matrix:")

print(cm)


# ------------------------------------------------------------
# 28. PLOT CONFUSION MATRIX
# ------------------------------------------------------------

plt.figure(
    figsize=(6, 5)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Fake News Detection - Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    [0, 1],
    ["FAKE", "REAL"]
)

plt.yticks(
    [0, 1],
    ["FAKE", "REAL"]
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 29. PLOT TRAINING ACCURACY
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Training vs Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.grid()

plt.show()


# ------------------------------------------------------------
# 30. PLOT TRAINING LOSS
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Training vs Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.grid()

plt.show()


# ------------------------------------------------------------
# 31. SAVE MODEL
# ------------------------------------------------------------

MODEL_PATH = "fake_news_lstm_model.keras"

model.save(
    MODEL_PATH
)

print(
    f"\nModel saved as: {MODEL_PATH}"
)


# ------------------------------------------------------------
# 32. SAVE TOKENIZER
# ------------------------------------------------------------

TOKENIZER_PATH = "tokenizer.pkl"

with open(
    TOKENIZER_PATH,
    "wb"
) as file:

    pickle.dump(
        tokenizer,
        file
    )


print(
    f"Tokenizer saved as: {TOKENIZER_PATH}"
)


# ------------------------------------------------------------
# 33. FUNCTION FOR NEW PREDICTION
# ------------------------------------------------------------

def predict_news(news_text):

    # Clean input
    cleaned_text = clean_text(
        news_text
    )

    # Convert to sequence
    sequence = tokenizer.texts_to_sequences(
        [cleaned_text]
    )

    # Pad sequence
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    # Predict probability
    probability = model.predict(
        padded_sequence,
        verbose=0
    )[0][0]

    # Classification
    if probability >= 0.5:

        label = "REAL"

        confidence = probability * 100

    else:

        label = "FAKE"

        confidence = (1 - probability) * 100


    return label, confidence


# ------------------------------------------------------------
# 34. USER INPUT PREDICTION
# ------------------------------------------------------------

print("\n")
print("=" * 50)
print("FAKE NEWS DETECTION")
print("=" * 50)

while True:

    user_news = input(
        "\nEnter news article (or type 'exit' to quit):\n"
    )

    if user_news.lower() == "exit":

        print(
            "\nExiting program..."
        )

        break


    if user_news.strip() == "":

        print(
            "Please enter some news text."
        )

        continue


    prediction, confidence = predict_news(
        user_news
    )


    print("\n")
    print("-" * 50)

    print(
        "Prediction:",
        prediction
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

    print("-" * 50)
