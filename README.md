# Amazon Sentiment Analysis

End-to-end machine learning project for sentiment classification on Amazon product reviews using classical NLP techniques.

## Overview

This project demonstrates a complete machine learning workflow applied to real-world text data. The goal is to classify customer reviews as positive or negative sentiment and evaluate model performance using appropriate metrics beyond simple accuracy.

## Problem Statement

Online reviews contain valuable information about customer opinions.
The objective of this project is to build a machine learning model that can automatically determine the sentiment of Amazon product reviews.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

```bash
pip install -r requirements.txt
```

## Dataset

The dataset consists of Amazon customer reviews with associated sentiment labels.

## Approach

The project follows a standard machine learning pipeline:

- Data loading and basic exploration
- Text preprocessing (cleaning, normalization)
- Feature extraction using **TF-IDF**
- Model training using classical machine learning algorithms:
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machine (SVM)
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC

## Results

The trained models achieved strong performance on the sentiment classification task, with accuracy up to **93%** and a **ROC-AUC score of 0.96**.
Additional metrics such as precision and recall were used to ensure robust evaluation.

## Project Structure

```text
amazon-sentiment-analysis/
├── data/
│ └── amazon_reviews.csv
└── notebooks/
└── amazon_sentiment_analysis.ipynb

```

## Notebook

The full analysis and implementation can be found here:
📓 `notebooks/amazon_sentiment_analysis.ipynb`

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn

## Key Learnings

- Importance of proper text preprocessing in NLP tasks
- Trade-offs between different classical ML models
- Why evaluation metrics beyond accuracy are critical for reliable model assessment

## Future Improvements

- Compare classical models with Transformer-based approaches (e.g., BERT)
- Hyperparameter tuning
- Model deployment using an API

```

```
