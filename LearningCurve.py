import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

df = pd.read_csv(
    "..\Kursmaterialien\Abschnitt 26 - Entscheidungsbaeume\classification.csv")

print(df.head())

X = df[["age", "interest"]].values
y = df["success"].values

import numpy as np

X, y = shuffle(X, y)

train_sizes_abs, train_scores, test_scores = learning_curve(KNeighborsClassifier(), X, y)

import matplotlib.pyplot as plt

plt.plot(train_sizes_abs, np.mean(train_scores, axis = 1))
plt.plot(train_sizes_abs, np.mean(test_scores, axis = 1))

plt.show()
