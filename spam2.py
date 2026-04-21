from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

X = [
    [10, 5, 1], [1, 0, 0], [8, 3, 1], [0, 1, 0],
    [7, 4, 1], [1, 0, 0], [9, 6, 1], [0, 0, 0],
    [6, 3, 1], [2, 0, 0], [8, 5, 1], [0, 2, 0],
    [7, 3, 1], [1, 1, 0], [9, 4, 1], [0, 0, 0],
    [5, 2, 1], [2, 1, 0], [8, 6, 1], [1, 0, 0]
]

Y = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, Y_train)

predictions = model.predict(X_test)
print("Predictions:", predictions)
print("Actual:", Y_test)
print("Accuracy:", accuracy_score(Y_test, predictions))

cm = confusion_matrix(Y_test, predictions)
print("Confusion Matrix:")
print(cm)