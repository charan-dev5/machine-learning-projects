# Machine Learning



![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)




![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)




![Flask](https://img.shields.io/badge/Flask-REST%20API-black?logo=flask)




![Status](https://img.shields.io/badge/Status-Complete-brightgreen)



Machine learning projects built from scratch using Python and scikit-learn —
covering classification, regression, and live model deployment via Flask API.

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

### Random Forest Classifier (`trees.py`)
Predicts customer churn using ensemble learning.
- Model: RandomForestClassifier (100 estimators)
- Metric: Accuracy Score

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
- Train/Test Split + Overfitting vs Generalization
- Model Serialization (joblib/pickle)
- ML Model Deployment via Flask REST API

## Setup
pip install scikit-learn flask joblib

## Author
**Sai Charan** - Python Developer & AI/ML Specialist

[![GitHub](https://img.shields.io/badge/GitHub-charan--dev5-black?logo=github)](https://github.com/charan-dev5)

[![Fiverr](https://img.shields.io/badge/Fiverr-charan__dev5-1DBF73?logo=fiverr)](https://www.fiverr.com/charan_dev5)

