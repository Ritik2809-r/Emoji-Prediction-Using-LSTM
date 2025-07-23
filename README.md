# 📌 Emoji Prediction Using LSTM

## 🧠 Overview

This project implements a deep learning model using **LSTM (Long Short-Term Memory)** networks to predict emojis based on text input.

It's a deep learning model trained using **TensorFlow** and **Keras**. To improve performance, I used various text processing techniques such as:

- **Tokenizing words** (breaking sentences into smaller pieces)
- **Removing noise** from text inputs

I also integrated **pre-trained GloVe word embeddings** to help the model better understand the context of words.

Additionally, I applied techniques like **Dropout** and **Hyperparameter Tuning** to reduce overfitting and improve model accuracy.

---

## ⚙️ Model Architecture

- **Input Layer**: Text inputs are preprocessed and converted to GloVe word embeddings  
- **LSTM Layers**: A 2-layer LSTM architecture for capturing text sequence patterns  
- **Dropout Layer**: To prevent overfitting  
- **Dense Output Layer**: For categorical emoji prediction with softmax activation  

---

## 🚀 Key Features

- ✅ Converts raw text into numerical vectors using **GloVe embeddings**  
- ✅ Implements **multi-layer LSTM** for effective sequence learning  
- ✅ Supports **confidence score output** along with predictions  
- ✅ Includes **evaluation metrics** like accuracy, confusion matrix, etc.  
- ✅ Ready-to-use **interactive demo** for testing custom text inputs  


# 📌 Project Workflow – Emoji Prediction Using LSTM

This section explains the entire pipeline followed to build the emoji prediction model using LSTM.

---

## 1️⃣ Load & Prepare Data

The first step is to **load the training and test datasets** from CSV files and prepare emoji labels.

### 📂 Dataset

- `train_emoji.csv`: Contains training samples with text and corresponding emoji labels
- `test_emoji.csv`: Used to evaluate the trained model

```python
import pandas as pd

# Load datasets
train = pd.read_csv("train_emoji.csv")
test = pd.read_csv("test_emoji.csv")

# Emoji mapping dictionary
emoji_dict = {
    0: "❤️",   # red heart
    1: "🏀",   # basketball
    2: "😂",   # face with tears of joy
    3: "😕",   # confused face
    4: "🔪"    # knife
}

## 2️⃣ Text Preprocessing & Word Embedding

Before feeding the data into the LSTM model, text must be tokenized and converted into numerical vectors that the model can understand.

---

### 🧠 Word Embeddings (GloVe)

We use **GloVe** (Global Vectors for Word Representation), a pre-trained embedding technique that captures the **semantic meaning** and **context** of words.

---

### 🔢 Load GloVe Embeddings

```python
import numpy as np

# Load GloVe embeddings
embeddings_index = {}
with open("glove.6B.50d.txt", encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = vectors

## 3️⃣ Build, Compile & Train the LSTM Model

We define our LSTM-based neural network using the **Keras Sequential API**, ideal for stacked layers in a linear pipeline.

---

### 🧱 Architecture Overview

Our model is designed to capture sequential dependencies in text data:

- **LSTM Layer 1**: 64 units  
  Captures short-term dependencies in the input sequences.

- **Dropout**: 0.5  
  Reduces overfitting by randomly deactivating 50% of neurons during training.

- **LSTM Layer 2**: 128 units  
  Builds on the previous layer to learn higher-level patterns.

- **Dropout**: 0.5  
  Further regularization to enhance generalization.

- **Dense Output Layer**:  
  Fully connected layer with **Softmax activation** to output probabilities across **5 target classes**.

---

### 🧾 Model Definition (Minimal Code)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(max_len, 50)),
    Dropout(0.5),
    LSTM(128),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

## 4️⃣ Model Evaluation

Once the model is trained, it’s crucial to **evaluate its performance** on unseen data. We assess accuracy, precision, recall, and how well the model distinguishes between different emoji classes using a **classification report** and a **confusion matrix**.

---

### 🧾 Evaluation Code

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model():
    # Evaluate on train and test sets
    train_loss, train_acc = model.evaluate(X_train, y_train)
    test_loss, test_acc = model.evaluate(X_test, y_test)

    print(f"Train Accuracy: {train_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")

    # Predict class labels
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=list(emoji_dict.values())))

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=emoji_dict.values(),
                yticklabels=emoji_dict.values(),
                cmap="Blues")
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

## 5️⃣ Emoji Prediction Function

To make the model interactive, we create a **custom prediction function** that takes a user-entered sentence and returns the most likely emoji based on the learned context.

This is useful for applications like **chatbots**, **sentiment-based emoji suggestions**, or **social media automation**.

---

### 🧾 Prediction Code

```python
import emoji
import numpy as np

def predict_emoji(text):
    """Predict an emoji for the given input sentence."""
    
    # Convert input text into padded embedding
    embedding, _ = text_to_embeddings([text], embeddings_index, max_len=max_len)
    
    # Generate prediction probabilities
    prediction = model.predict(embedding)
    
    # Get the predicted class and its confidence score
    pred_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # Print input and predicted emoji
    print(f"\nInput: '{text}'")
    print(f"Predicted Emoji: {emoji.emojize(emoji_dict[pred_class])} (Confidence: {confidence*100:.1f}%)")

    # Print all class probabilities
    print("\nAll Probabilities:")
    for i, prob in enumerate(prediction[0]):
        print(f"{emoji.emojize(emoji_dict[i])}: {prob*100:.1f}%")

## 6️⃣ Interactive Emoji Predictor

This simple command-line interface allows users to **test the model in real-time** by typing in custom sentences. It’s ideal for quick validation, demos, or user feedback.

---

### 🧾 Interactive CLI Code

```python
print("\n🔁 Interactive Emoji Predictor")
print("Type 'quit' to exit.")

while True:
    user_input = input("\nEnter text to predict emoji: ")
    if user_input.lower() == 'quit':
        print("👋 Exiting Emoji Predictor.")
        break
    predict_emoji(user_input)
