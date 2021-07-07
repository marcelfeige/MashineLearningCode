import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

df = pd.read_csv("D:\Eigene Datein\eLearning Kurse\Machine Learning\Kursmaterialien"
                 "\Abschnitt 08 - Lineare Regression mit mehreren Variablen\hotels.csv")

X = df[["Gewinn", "Quadratmeter"]].values
Y = df[["Preis in Mio"]].values

kf = KFold(n_splits = 3, shuffle = True)
kf.split(X)

for train_index, test_index in kf.split(X):
    print("train: " + str(train_index))
    print("test: " + str(test_index))

    X_test = X[test_index]
    X_train = X[train_index]

    y_test = Y[test_index]
    y_train = Y[train_index]

    model = LinearRegression()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print("Score: " + str(score))
    print("---------")

# Kurzschreibweise
scores = cross_val_score(LinearRegression(), X, Y, cv = KFold(n_splits = 6))
print(scores)
print("Score Mittelwert (Kuzrschreibweise): " + str(np.mean(scores)))


