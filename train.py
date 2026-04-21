from sklearn.linear_model import LogisticRegression
import joblib
                                                    #trains and saves model.pkl
X = [
    [10, 5, 1],    #1
    [1, 0, 0],     #0
    [8, 3, 1],     #1
    [0, 1, 0],     #0                                    
    [7, 4, 1],     #1
    [1, 0, 0],     #0
    [9, 6, 1],     #1
    [0, 0, 0]      #0

]
Y = [1, 0, 1, 0, 1, 0, 1, 0]

model = LogisticRegression()
model.fit(X, Y)

joblib.dump(model, "model.pkl")
print("Model saved.")