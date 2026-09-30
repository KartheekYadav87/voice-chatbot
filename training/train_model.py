import json
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "intents.json")
MODEL_DIR = os.path.join(BASE_DIR, "..", "backend", "models")

os.makedirs(MODEL_DIR, exist_ok=True)

with open(DATA_PATH, "r") as f:
    data = json.load(f)

sentences = []
labels = []
responses_map = {}

for intent in data["intents"]:
    tag = intent["tag"]
    responses_map[tag] = intent["responses"]
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(tag)

print(f"Total training samples: {len(sentences)}")
print(f"Total intents: {len(set(labels))}")

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

VOCAB_SIZE = 1000
MAX_LEN = 20
EMBEDDING_DIM = 64

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>", lower=True)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post")
y = encoded_labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_classes = len(label_encoder.classes_)

model = keras.Sequential([
    layers.Input(shape=(MAX_LEN,)),
    layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, mask_zero=True),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=15,
    restore_best_weights=True,
    mode="max"
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=8,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTraining Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

model_path = os.path.join(MODEL_DIR, "chatbot_model.keras")
model.save(model_path)
print(f"Model saved to: {model_path}")

tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.pkl")
with open(tokenizer_path, "wb") as f:
    pickle.dump(tokenizer, f)
print(f"Tokenizer saved to: {tokenizer_path}")

label_encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
with open(label_encoder_path, "wb") as f:
    pickle.dump(label_encoder, f)
print(f"Label encoder saved to: {label_encoder_path}")

responses_path = os.path.join(MODEL_DIR, "responses.json")
with open(responses_path, "w") as f:
    json.dump(responses_map, f)
print(f"Responses saved to: {responses_path}")

config = {
    "vocab_size": VOCAB_SIZE,
    "max_len": MAX_LEN,
    "embedding_dim": EMBEDDING_DIM,
    "num_classes": num_classes,
    "labels": label_encoder.classes_.tolist(),
    "training_accuracy": float(train_acc),
    "test_accuracy": float(test_acc)
}
config_path = os.path.join(MODEL_DIR, "config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"Config saved to: {config_path}")

print("\n--- Model Training Complete ---")
print(f"Intents covered: {', '.join(label_encoder.classes_)}")
print(f"Model ready for deployment!")
