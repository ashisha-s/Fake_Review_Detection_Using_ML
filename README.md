# 🔍 Fake Review Detection Using Machine Learning

### Detecting Deceptive Online Reviews Using NLP and Machine Learning

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Text%20Classification-blueviolet?style=flat-square)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-green?style=flat-square)

---

## 📌 Project Overview

Online reviews play an important role in influencing customer decisions. However, deceptive reviews can mislead consumers, manipulate business reputations, and reduce trust in online platforms.

**Fake Review Detection Using Machine Learning** is a Natural Language Processing (NLP) based classification project designed to identify whether a hotel review is genuine or deceptive.

The system analyzes textual content using preprocessing techniques, linguistic characteristics, sentiment analysis, and TF-IDF feature extraction.

Three supervised machine learning algorithms are considered:

- Logistic Regression
- Linear Support Vector Machine (Linear SVM)
- Random Forest

Their performance is compared using standard classification evaluation metrics.

## 🎯 Objectives

- Identify deceptive and truthful hotel reviews using machine learning.
- Apply NLP techniques to clean and normalize review text.
- Extract meaningful textual, linguistic, and sentiment-based features.
- Train and compare multiple supervised classification algorithms.
- Evaluate model performance using standard classification metrics.
- Develop a system capable of predicting the authenticity of review text.

---

## 📂 Dataset

### Ott et al. Gold Standard Deceptive Opinion Spam Dataset

The project uses the **Deceptive Opinion Spam Corpus**, developed by Ott et al., containing hotel reviews associated with hotels in Chicago.

The dataset is widely used in research on deceptive opinion detection and text classification.

| Property | Description |
|---|---|
| Dataset | Ott et al. Deceptive Opinion Spam Corpus |
| Domain | Hotel Reviews |
| Location | Chicago, USA |
| Total Reviews | 1,600 |
| Truthful Reviews | 800 |
| Deceptive Reviews | 800 |
| Classification | Binary |
| Data Type | Text |

The dataset provides balanced examples of truthful and deceptive reviews, making it suitable for supervised classification experiments.

**Dataset Source:** [Deceptive Opinion Spam Corpus – Ott et al.](https://myleott.com/op-spam.html)

**Note:** The deceptive reviews in this corpus are intentionally written for research. They do not represent every type of fraudulent or AI-generated review encountered in real-world platforms.

---

## ⚙️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Natural Language Processing | NLTK |
| Feature Extraction | TF-IDF, Linguistic Features |
| Sentiment Analysis | VADER |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Development Environment | Google Colab, VS Code |
| Version Control | Git, GitHub |

---

## 🔄 Project Workflow

The fake review detection process follows a structured machine learning pipeline.

**1. Dataset Collection**

↓  

**2. Exploratory Data Analysis**

↓  

**3. Text Preprocessing**

↓  

**4. Feature Extraction**

↓  

**5. Feature Selection and Preparation**

↓  

**6. Model Training**

↓  

**7. Model Evaluation and Comparison**

↓  

**8. Fake / Genuine Review Prediction**

---

## 🧹 Data Preprocessing

Raw review text is transformed into a cleaner and more consistent representation before feature extraction.

The preprocessing operations include:

1. **Lowercase Conversion:** Converts text into lowercase for consistency.
2. **Special Character Removal:** Removes unnecessary punctuation and special symbols.
3. **Tokenization:** Splits review text into individual words or tokens.
4. **Stopword Removal:** Removes frequently occurring words that contribute limited contextual information.
5. **Lemmatization:** Converts words into their base or dictionary forms.

These operations help reduce textual noise and improve feature consistency.

---

## 🧠 Feature Extraction

Multiple feature categories are used to capture patterns associated with deceptive and truthful reviews.

### 1. TF-IDF Features

Term Frequency–Inverse Document Frequency (TF-IDF) transforms review text into numerical vectors based on the importance of words within the document collection.

TF-IDF helps machine learning models identify discriminative textual patterns.

### 2. Linguistic Features

Linguistic features capture the structural and writing characteristics of reviews.

Features include:

- Review Length
- Word Count
- Sentence Count
- Average Word Length
- Exclamation Count
- Question Count
- Capital Letter Count

### 3. Sentiment Features

Sentiment analysis is performed using **VADER (Valence Aware Dictionary and sEntiment Reasoner)**.

The following sentiment scores are extracted:

- Positive Score
- Negative Score
- Neutral Score
- Compound Score

These features provide additional information about the emotional tone of review text.

---

## 🤖 Machine Learning Algorithms

### Logistic Regression

Logistic Regression is a supervised classification algorithm that estimates the probability of a review belonging to a particular class.

It serves as an interpretable baseline for text classification.

### Linear Support Vector Machine (Linear SVM)

Linear SVM identifies a separating hyperplane between deceptive and truthful review classes.

It is particularly suitable for high-dimensional, sparse feature representations such as TF-IDF.

### Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to generate classification predictions.

It is used to investigate the effectiveness of tree-based classification for the extracted features.

---

## 📊 Model Evaluation

The machine learning models are evaluated using the following performance metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall classification correctness |
| Precision | Measures the reliability of positive predictions |
| Recall | Measures the ability to identify positive examples |
| F1-Score | Balances precision and recall |
| ROC-AUC | Measures discrimination between the two classes |
| Confusion Matrix | Visualizes correct and incorrect classifications |

The models are compared to identify the most suitable classification approach for the selected dataset.

---

## 🔎 Review Prediction

The prediction component accepts review text and processes it using the same preprocessing and feature extraction pipeline used during model development.

The trained classifier then predicts one of two categories:

**Genuine Review:** The model predicts that the review belongs to the truthful class.

**Fake Review:** The model predicts that the review belongs to the deceptive class.

Predictions represent statistical classifications and should not be treated as definitive proof of review authenticity.

---

## 🚀 Future Enhancements

- Improve classification performance through hyperparameter optimization.
- Investigate additional linguistic and contextual features.
- Extend the system to reviews from multiple domains.
- Improve model interpretability.
- Deploy the classifier as a web application for interactive review analysis.

---

## 📚 References

**Ott, M., Choi, Y., Cardie, C., & Hancock, J. T. (2011).** Finding Deceptive Opinion Spam by Any Stretch of the Imagination. *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics.*

**Ott, M., Cardie, C., & Hancock, J. T. (2013).** Negative Deceptive Opinion Spam. *Proceedings of NAACL-HLT.*

Dataset: https://myleott.com/op-spam.html

---

## 👩‍💻 Author

**Ashisha S**

Master of Computer Applications (MCA)  
Mar Athanasius College of Engineering

[![GitHub](https://img.shields.io/badge/GitHub-ashisha--s-181717?style=flat-square&logo=github)](https://github.com/ashisha-s)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/ashisha-s)

---

⭐ *If you find this project interesting, consider giving the repository a star!*
