# Machine Learning



![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)




![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)




![Flask](https://img.shields.io/badge/Flask-REST%20API-black?logo=flask)




![Status](https://img.shields.io/badge/Status-Complete-brightgreen)



Machine learning projects built from scratch using Python and scikit-learn —
covering classification, regression, and live model deployment via Flask API.

## Projects

### House Price Predictor (`house.py`)
Predicts house prices using Random Forest Regression.
- Dataset: California Housing (sklearn built-in)
- Model: RandomForestRegressor
- Metrics: MAE, R2 Score

### Spam Classifier (`spam.py`)
Classifies messages as spam or not spam.
- Model: RandomForestClassifier
- Metric: Accuracy Score
- Saved model: `spam_model.pkl`

### Ensemble Explorer (`trees.py`)
Explores sklearn ensemble methods including
GradientBoosting and AdaBoost regressors.

### Logistic Regression Trainer (`train.py`)
Binary classification using Logistic Regression
with joblib model serialization.

### ML Prediction API (`app.py`)
Flask REST API exposing a `/predict` endpoint
that loads a trained model and returns live predictions.

## Tech Stack
- Python 3.14
- scikit-learn
- pandas
- Flask
- joblib

## Concepts Covered
- Classification, Regression, Ensemble Methods
- Train/Test Split + Overfitting vs Generalization
- Model Serialization (joblib/pickle)
- ML Model Deployment via Flask REST API

## Setup
pip install scikit-learn pandas flask joblib

## Author
**Sai Charan** - Python Developer & AI/ML Specialist 

[![GitHub](https://img.shields.io/badge/GitHub-charan--dev5-black?logo=github)](https://github.com/charan-dev5)

[![Fiverr](https://img.shields.io/badge/Fiverr-charan__dev5-1DBF73?logo=fiverr)](https://www.fiverr.com/charan_dev5)
