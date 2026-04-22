# 🛍️ Amazon Sentiment Analysis API

End-to-end MLOps pipeline — from raw data to a production-ready REST API with MLflow experiment tracking, CI/CD, and Docker containerization. Achieves CV F1 0.858 · ROC-AUC 0.967.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green)
![MLflow](https://img.shields.io/badge/MLflow-3.1.1-orange)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![CI](https://github.com/ozairshafique/amazon-sentiment-analysis/actions/workflows/ci.yml/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen)
![Tests](https://img.shields.io/badge/tests-20%2B%20passed-brightgreen)

## Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Approach](#-approach)
- [Results](#-results)
- [MLflow Experiment Tracking](#-mlflow-experiment-tracking)
- [Visualizations](#-visualizations)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [API Endpoints](#-api-endpoints)
- [Testing](#-testing)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Reproducibility](#-reproducibility)
- [Key Learnings](#-key-learnings)
- [Future Improvements](#-future-improvements)
- [Conclusion](#-conclusion)
- [Author](#-author)

---

## 📌 Overview

This project demonstrates a complete machine learning lifecycle:

- Data exploration and model comparison in Jupyter
- Clean training pipeline with GridSearchCV
- **MLflow experiment tracking** — parameters, metrics, and artifacts logged per run
- Production REST API built with FastAPI
- Containerized with Docker
- Automated CI/CD with GitHub Actions
- 20+ pytest tests with **98% coverage**

---

## 🎯 Problem Statement

Online reviews contain valuable information about customer opinions, but analyzing them manually is not scalable.

The objective is to build a machine learning model that automatically classifies Amazon product reviews as **positive** or **negative** sentiment.

---

## 📂 Dataset

The dataset consists of Amazon customer reviews with associated star ratings used to derive sentiment labels.

| Stars                | Label    | Encoded |
| -------------------- | -------- | ------- |
| ⭐⭐⭐⭐⭐ 4–5 stars | Positive | 1       |
| ⭐ 1–2 stars         | Negative | 0       |
| ⭐⭐⭐ 3 stars       | Neutral  | Dropped |

| Sentiment           | Count      | Ratio |
| ------------------- | ---------- | ----- |
| Positive            | 4,448      | 93%   |
| Negative            | 324        | 7%    |
| **Imbalance ratio** | **13.7:1** |       |

> Download the dataset from Kaggle and place it as `data/amazon_reviews.csv`
> 🔗 [Amazon Product Reviews – Kaggle](https://www.kaggle.com/datasets/halimedogan/amazon-reviews)

---

## 🚀 Approach

![Workflow](images/workflows.png)

---

## 📊 Results

### Model Comparison

| Metric             | Logistic Regression |             LinearSVC |         SVC |    Random Forest |
| ------------------ | ------------------: | --------------------: | ----------: | ---------------: |
| Best C             |                  10 |                   0.1 |           — |                — |
| CV Macro F1        |           **0.858** |                 0.850 |       0.840 |            0.512 |
| Test F1 (Macro)    |           **0.852** |                 0.835 |           — |                — |
| Test F1 (Positive) |           **0.980** |                 0.975 |           — |                — |
| Test F1 (Negative) |           **0.724** |                 0.694 |           — |                — |
| ROC-AUC            |               0.967 |             **0.968** |       0.968 |            0.961 |
| Production         |     ✅ **Selected** | ❌ No `predict_proba` | ❌ Too slow | ❌ Baseline only |

### Why Logistic Regression?

| Criterion        | LogReg    | LinearSVC         | SVC        | Random Forest |
| ---------------- | --------- | ----------------- | ---------- | ------------- |
| CV Macro F1      | ✅ Best   | ✅ Similar        | ✅ Similar | ❌ Lowest     |
| predict_proba()  | ✅ Native | ❌ Wrapper needed | ⚠️ Slow    | ✅ Native     |
| Inference speed  | ✅ Fast   | ✅ Fast           | ⚠️ Slow    | ❌ Slow       |
| Production ready | ✅        | ❌                | ❌         | ❌            |

> ⚠️ Accuracy alone is misleading with 13.7:1 class imbalance.
> **F1 macro** is used as the primary evaluation metric.

---

## 🔬 MLflow Experiment Tracking

All training runs are tracked with **MLflow 3.1.1**, enabling full reproducibility and side-by-side run comparison.

![MLflow Run Comparison](images/model_comparisons.jpeg)

### What is logged per run

**Parameters**

| Parameter                       | Description                                    |
| ------------------------------- | ---------------------------------------------- |
| `model`                         | Algorithm name (`logreg`, `linear_svc`, etc.)  |
| `best_C`                        | Best regularization strength from GridSearchCV |
| `cv_folds`                      | Number of cross-validation folds (3)           |
| `cv_scoring`                    | Scoring metric used (`f1_macro`)               |
| `tfidf_max_features`            | Maximum TF-IDF vocabulary size (20,000)        |
| `tfidf_ngram_range`             | N-gram range `(1, 3)`                          |
| `tfidf_sublinear_tf`            | Log-frequency scaling (`True`)                 |
| `tfidf_max_df` / `tfidf_min_df` | Document frequency bounds                      |
| `test_size`                     | Train/test split ratio (0.2)                   |
| `random_seed`                   | Global random seed (42)                        |

**Metrics**

| Metric             | Logistic Regression | LinearSVC |
| ------------------ | ------------------: | --------: |
| `best_cv_f1`       |           **0.858** |     0.850 |
| `test_f1_macro`    |           **0.852** |     0.835 |
| `test_f1_negative` |           **0.724** |     0.694 |
| `test_f1_positive` |           **0.980** |     0.975 |
| `test_roc_auc`     |               0.967 | **0.968** |

**Artifacts:** confusion matrix, classification report, trained model (`.pkl`)

### Launch the MLflow UI

```bash
mlflow ui
# http://localhost:5000
```

---

## 📦 Model Registry

The production model is registered and versioned in MLflow Model Registry.

![Model Regiestered](images/model-regesterd.jpeg)

| Model                   | Version | Stage         | CV F1 Macro |
| ----------------------- | ------- | ------------- | ----------- |
| amazon-sentiment-logreg | 2       | ✅ Production | 0.858       |
| amazon-sentiment-logreg | 1       | 📦 Archived   | —           |

> In a production cloud environment the API would load directly via
> `models:/amazon-sentiment-logreg/Production` against a remote tracking server.

---

## 📈 Visualizations

### Confusion Matrix

![Confusion Matrix](images/confusion_matrices.png)

### ROC Curve

![ROC Curve](images/roc_curves.png)

### Class Distribution

![Class Distribution](images/class_distribution.png)

### Cross-Validated Macro F1

![CV F1](images/cv_f1_comparison.png)

---

## 🛠️ Tech Stack

| Category            | Tools                       |
| ------------------- | --------------------------- |
| Language            | Python 3.12                 |
| ML                  | Scikit-learn, NumPy, Pandas |
| NLP                 | NLTK, TextBlob              |
| API                 | FastAPI, Uvicorn, Pydantic  |
| Experiment Tracking | MLflow 3.1.1                |
| Testing             | Pytest, pytest-cov          |
| Containerization    | Docker                      |
| CI/CD               | GitHub Actions              |
| Visualization       | Matplotlib, Seaborn         |

---

## 📂 Project Structure

```
amazon-sentiment-analysis/
│
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI/CD
│
├── src/
│   ├── preprocessing.py              # TextPreprocessor (sklearn-compatible)
│   ├── model_loader.py               # Lazy model loading & inference
│   └── train.py                      # Training pipeline — GridSearchCV + MLflow logging
│
├── api/
│   ├── __init__.py
│   ├── main.py                       # FastAPI application
│   └── schemas.py                    # Pydantic request/response schemas
│
├── tests/
│   ├── test_preprocessing.py         # 100% coverage
│   ├── test_model.py                 # 92% coverage
│   └── test_api.py
│
├── notebooks/
│   └── amazon_sentiment_analysis.ipynb
│
├── mlruns/                           # MLflow experiment tracking (git-ignored)
├── model/                            # Saved .pkl files (git-ignored)
├── data/                             # Dataset CSV (git-ignored)
├── images/                           # Visualizations
│
├── Dockerfile
├── requirements.txt
├── pytest.ini
├── .gitignore
└── README.md
```

---

## ▶️ How to Run

### Option 1 — Local

```bash
# 1. Clone the repository
git clone https://github.com/ozairshafique/amazon-sentiment-analysis.git
cd amazon-sentiment-analysis

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -m nltk.downloader stopwords wordnet

# 5. Download dataset from Kaggle and place at:
#    data/amazon_reviews.csv

# 6. Train the model (auto-logs run to MLflow)
python src/train.py

# 7. (Optional) Inspect experiments
mlflow ui                       # http://localhost:5000

# 8. Start the API
uvicorn api.main:app --reload   # http://localhost:8000
```

### Option 2 — Docker

```bash
# Build image
docker build -t amazon-sentiment-analysis .

# Run container
docker run -p 8000:8000 amazon-sentiment-analysis
```

---

## 🌐 API Endpoints

| Method | Endpoint         | Description                |
| ------ | ---------------- | -------------------------- |
| GET    | `/`              | API info                   |
| GET    | `/health`        | Health check               |
| POST   | `/predict`       | Single prediction          |
| POST   | `/predict/batch` | Batch prediction (max 100) |

### Single Prediction Example

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d "{\"text\": \"This product is amazing!\", \"model_name\": \"logreg\"}"
```

### Response

```json
{
  "text": "This product is amazing!",
  "model": "logreg",
  "sentiment": "positive",
  "confidence": 0.9808
}
```

### Batch Prediction Example

```json
[
  { "text": "Amazing product!", "model_name": "logreg" },
  { "text": "Terrible quality!", "model_name": "logreg" },
  { "text": "Great value for money!", "model_name": "logreg" }
]
```

### Batch Response

```json
{
  "total": 3,
  "results": [
    {
      "text": "Amazing product!",
      "model": "logreg",
      "sentiment": "positive",
      "confidence": 0.98
    },
    {
      "text": "Terrible quality!",
      "model": "logreg",
      "sentiment": "negative",
      "confidence": 0.75
    },
    {
      "text": "Great value for money!",
      "model": "logreg",
      "sentiment": "positive",
      "confidence": 0.99
    }
  ]
}
```

> 📖 Interactive API docs available at: `http://localhost:8000/docs`

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Coverage Report

| File               | Coverage |
| ------------------ | -------- |
| `preprocessing.py` | 100% ✅  |
| `model_loader.py`  | 83% ✅   |
| `train.py`         | 99% ✅   |
| **Total**          | **98%**  |

> 20 tests covering preprocessing, model inference, and API endpoints.

---

## ⚙️ CI/CD Pipeline

Every push to `main` automatically triggers:

1. **Environment setup** — Python 3.12, install all dependencies
2. **NLTK data** — download stopwords and WordNet
3. **Training** — runs `src/train.py`, logs run to MLflow (`mlruns/`)
4. **Testing** — 20+ pytest tests, enforces 97% coverage
5. **Docker build** — builds and validates the container image

---

## 🔁 Reproducibility

All experiments are fully reproducible:

| Factor           | Value                    |
| ---------------- | ------------------------ |
| Random seed      | 42 (globally fixed)      |
| Train/test split | 80/20, stratified        |
| CV strategy      | 3-fold, stratified       |
| CV scoring       | F1 macro                 |
| Tracked by       | MLflow 3.1.1 (`mlruns/`) |

```bash
# Re-run training and compare with previous runs
python src/train.py
mlflow ui              # view all runs at localhost:5000
```

---

## 🧠 Key Learnings

- Why accuracy is misleading with imbalanced datasets — and how F1 macro addresses it
- How sklearn Pipelines prevent data leakage
- Production model selection based on multiple criteria, not just a single metric
- MLflow experiment tracking for reproducible, comparable model runs
- FastAPI best practices for ML APIs
- Docker containerization for ML applications
- CI/CD automation with GitHub Actions

---

## 🔮 Future Improvements

- [ ] Fine-tune DistilBERT for contextual embeddings
- [ ] Add LIME/SHAP for prediction explainability
- [ ] Add Prometheus + Grafana for API monitoring
- [ ] Implement data drift detection with Evidently AI
- [ ] Collect more negative reviews to reduce 13.7:1 class imbalance
- [ ] Deploy to Railway or Render

---

## 🏁 Conclusion

This project demonstrates the complete MLOps lifecycle — from raw data to a production-ready, monitored, and reproducible system.

**Logistic Regression** was selected as the production model, achieving:

- **CV Macro F1: 0.858**
- **Test F1 Macro: 0.852**
- **ROC-AUC: 0.967**

The system includes MLflow experiment tracking, a FastAPI inference API, Docker containerization, 98% test coverage, and a fully automated GitHub Actions CI/CD pipeline.

---

## 👤 Author

**Uzair Shafique**

- GitHub: [ozairshafique](https://github.com/ozairshafique)
- Email: uzair_11@hotmail.com
