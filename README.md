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
