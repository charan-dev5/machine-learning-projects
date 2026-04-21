# Machine Learning



![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)




![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)




![Flask](https://img.shields.io/badge/Flask-REST%20API-black?logo=flask)




![Status](https://img.shields.io/badge/Status-Complete-brightgreen)



Machine learning projects built from scratch using Python and scikit-learn —
covering classification, regression, clustering, and live model deployment via Flask API.

## Projects

### House Price Predictor (`house.py`)
Predicts house prices based on size using Linear Regression.
- Model: LinearRegression
- Metrics: R2 Score, MAE

### Spam Classifier (`spam.py`)
Classifies messages as spam or not spam using Logistic Regression.
- Model: LogisticRegression
- Metric: Accuracy Score
- Saved model: `spam_model.pkl`

### Spam Classifier v2 (`spam2.py`)
Extended spam classifier with larger dataset for proper confusion matrix analysis.
- Model: LogisticRegression
- Metrics: Accuracy Score, Confusion Matrix (TP, TN, FP, FN)

### Random Forest Classifier (`trees.py`)
Predicts customer churn using ensemble learning.
- Model: RandomForestClassifier (100 estimators)
- Metric: Accuracy Score

### Customer Churn Predictor (`churn.py`)
Predicts whether a customer will leave based on age, spending, and support calls.
- Data: `churn.csv` loaded with pandas
- Model: RandomForestClassifier (100 estimators)
- Metrics: Accuracy Score, Confusion Matrix
- Saved model: `churn_model.pkl`

### K-Means Clustering (`cluster.py`)
Groups customers into segments based on spending and visit frequency.
- Model: KMeans (n_clusters=3)
- Output: Cluster labels and cluster centers
- Type: Unsupervised learning (no labels)

### Model Trainer (`train.py`)
Trains the spam classifier and serializes it to disk.
- Model: LogisticRegression
- Serialization: joblib → `model.pkl`

### ML Prediction API (`app.py`)
Flask REST API exposing a `/predict` endpoint
that loads a trained model and returns live predictions.

## Tech Stack
- Python 3.14
- scikit-learn
- pandas
- Flask
- joblib

## How It Works
1. Run `train.py` to train and save the model
2. Run `app.py` to start the prediction API
3. Send a POST request to `/predict` with features

## Example Request
{"features": [10, 5, 1]}

## Example Response
{"prediction": 1}

## Concepts Covered
- Classification, Regression, Ensemble Methods
- Unsupervised Learning — K-Means Clustering
- Train/Test Split + Overfitting vs Generalization
- Model Evaluation — R2, MAE, Accuracy, Confusion Matrix
- Model Serialization (joblib/pickle)
- ML Model Deployment via Flask REST API

## Setup
pip install scikit-learn pandas flask joblib

## Author
**Sai Charan** - Python Developer & AI/ML Specialist

[![GitHub](https://img.shields.io/badge/GitHub-charan--dev5-black?logo=github)](https://github.com/charan-dev5)

[![Fiverr](https://img.shields.io/badge/Fiverr-charan__dev5-1DBF73?logo=fiverr)](https://www.fiverr.com/charan_dev5)

