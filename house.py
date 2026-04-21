from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

X = [[600], [800], [1000], [1200], [1500], [1800], [2000], [2200]]   #define x and y , feed in input x and predict y

Y = [150, 200, 250, 300, 370, 440, 500, 550]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) #split

model = LinearRegression()                       #train
model.fit(X_train, Y_train)

predictions = model.predict(X_test)              #test
print("Predictions:", predictions)
print("Actual:", Y_test)

score = model.score(X_test, Y_test)
mae = mean_absolute_error(Y_test, predictions)     
r2 = r2_score(Y_test, predictions)

print(f"Accuracy (R2 score): {score: .2f}")   #how good the predictions are r2 = 1.0 is good , 0.8 is ok  
print(f"Mean Absolute Error: {mae: .2f}")