import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# load data from CSV
df = pd.read_csv("churn.csv")

# separate features and target
X = df.drop("churned", axis=1)
Y = df["churned"]

# split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# train
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, Y_train)

# predict
predicitons = model.predict(X_test)

# evaluate
print("Accuracy:", accuracy_score(Y_test, predicitons))
print("Confusion Matrix:")
print(confusion_matrix(Y_test, predicitons))

# save
joblib.dump(model, "churn_model.pkl")
print("Model saved.")