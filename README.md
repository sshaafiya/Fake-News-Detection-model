# 📰 Fake News Detection Model using TensorFlow

## 📌 Objective
Build a deep learning model using TensorFlow to detect whether a news article is **FAKE** or **REAL** based on its textual content.

## 📂 Dataset
- **Name**: Fake and Real News Dataset  
- **Source**: [Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)  
- **Columns**: `title`, `text`, `subject`, `date`, `label`  
- **Target Label**: `label` (FAKE or REAL)

## ✅ Project Goals

### 1. Importing Libraries and Dataset
- Use essential libraries: NumPy, Pandas, TensorFlow, Scikit-learn
- Load the dataset and explore its structure

### 2. Preprocessing Dataset
- Clean the text data (remove punctuation, special characters, and stopwords)
- Convert labels (`FAKE`, `REAL`) into binary values (`0`, `1`)
- Tokenize and pad the news text
- Split into training and testing sets (80% - 20%)

### 3. Generating Word Embeddings
- Use `Tokenizer` and `pad_sequences` to create numerical representations of the text
- Limit vocabulary size and sequence length to improve performance

### 4. Model Architecture
- Build a Sequential Model in TensorFlow:
  - Embedding Layer
  - LSTM Layer
  - Dense Layer
  - Output Layer with Sigmoid Activation
- Compile with Binary Crossentropy loss and Adam optimizer

### 5. Model Evaluation and Prediction
- Evaluate the model on test data using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Predict on new/unseen articles

## 🧠 Model Summary

```python
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)      (None, 200, 64)            320000    
 lstm (LSTM)                (None, 64)                 33024     
 dropout (Dropout)          (None, 64)                 0         
 dense (Dense)              (None, 32)                 2080      
 output (Dense)             (None, 1)                  33        
=================================================================
Total params: 355,137
Trainable params: 355,137

📁 Folder Structure
Fake-News-Detection/
│
├── dataset/
│   └── fake_or_real_news.csv
│
├── notebook/
│   └── Fake_News_Detection_LSTM.ipynb
│
├── README.md
└── requirements.txt
