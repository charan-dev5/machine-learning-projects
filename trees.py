from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = [
    [25, 30000, 1],
    [45, 80000, 8],
    [30, 45000, 2],
    [50, 90000, 10],
    [22, 25000, 1],
    [40, 70000, 6],
    [35, 55000, 4],
    [60, 95000, 12]
]

Y = [1, 0, 1, 0, 1, 0, 0, 0]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, Y_train)

predictions = model.predict(X_test)
print("Predictions:", predictions)
print("Actual:", Y_test)
print("Accuracy:", accuracy_score(Y_test, predictions))