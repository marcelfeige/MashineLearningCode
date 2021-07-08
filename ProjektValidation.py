import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier
import numpy as np

df = pd.read_csv(
    "D:\Eigene Datein\eLearning Kurse\Machine Learning\Kursmaterialien\Abschnitt 26 - Entscheidungsbaeume\classification.csv")

df.head()

X = df[["age", "interest"]].values

y = df["success"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.25)

param_range = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

train_scores, test_scores = validation_curve(
    DecisionTreeClassifier(),
    X,
    y,
    param_name = "max_depth",
    param_range = param_range
)

plt.plot(param_range, np.mean(train_scores, axis = 1))
plt.plot(param_range, np.mean(test_scores, axis = 1))

plt.show()