# 📰 Sarcasm Detection with DistilBERT  

This project explores sarcasm detection in news headlines using both **traditional ML models** and **deep learning architectures**, culminating in a fine-tuned **DistilBERT** model.  
It also includes a **Streamlit app** for interactive testing of headlines.  

---

## 📖 Project Overview

 - This project focuses on sarcasm detection in news headlines, a challenging NLP task where context and subtle language cues determine meaning.

### I built and compared multiple approaches:

 1- Classical ML Models

   - Logistic Regression, Naive Bayes, SVM

   - Served as baselines for text classification

 2- Deep Learning Models

   - LSTM & GRU with word embeddings

   - Captured sequential dependencies in text

 3- Transformer-based Model

   - Fine-tuned DistilBERT from Hugging Face

   - Achieved the highest accuracy (93.1%)

### 🔑 Key Features

 - End-to-End Pipeline: Preprocessing, model training, evaluation

 - Model Comparison: Clear benchmarks across ML, DL, and Transformers

 - Interactive App: Streamlit-based web app for real-time sarcasm detection

 - Hosted Model: DistilBERT model uploaded to Hugging Face Hub for easy reuse

### 🎯 Outcomes

 - Showed how transformers outperform both traditional and RNN-based models in NLP classification tasks

 - Delivered a professional, deployable solution with a web interface and hosted model

### 💡 This project demonstrates the journey from baseline ML → deep learning → state-of-the-art transformers,






---

## 📊 Model Performance

I compared several models on the [News Headlines Sarcasm Detection Dataset](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection):

| Model                | Accuracy |
|-----------------------|----------|
| **DistilBERT (ours)** | **93.1%** |
| GRU                  | 85.3% |
| LSTM                 | 84.6% |
| Logistic Regression  | 83.4% |
| SVM                  | 82.9% |
| Naive Bayes          | 82.7% |

✅ DistilBERT significantly outperformed baseline models.  


---

## 🚀 Usage

### 1. Clone the repo
```bash
git clone https://github.com/YamenRM/NLP-Sarcasm-Detection-Project.git
cd NLP-Sarcasm-Detection-Project
```
### 2. Install dependencies
```
pip install -r requirements.txt

```
### 3. Run the Streamlit app
```
streamlit run app.py

```

---

## 🤗 Hugging Face Model
The fine-tuned DistilBERT model is hosted on Hugging Face Hub:
https://huggingface.co/YamenRM/sarcasm_model

### Quick Usage
```
from transformers import pipeline

classifier = pipeline("text-classification", model="YamenRM/sarcasm_model")

print(classifier("Oh great, another Monday morning meeting!"))
# [{'label': 'SARCASTIC', 'score': 0.93}]

```

---

## ✨ Author

 - YamenRM
 - Palestine|GAZA
 - 3RD year at UP

 - Stay strong 💪
