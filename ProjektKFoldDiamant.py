import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

df = pd.read_csv("..\Kursmaterialien\Abschnitt 13 - Praxis Polynomiale Regression\diamonds.csv")

print(df.head())

X_carat = df[["carat"]].values
X_LBH = df[["x", "y", "z"]].values
Y = df[["price"]]

# Lin Regression auf Basis Carat
scores_carat = cross_val_score(LinearRegression(), X_carat, Y, cv = RepeatedKFold(n_repeats = 10))
# print("Scores Carat : " + str(scores_carat))
mean_carat = np.mean(scores_carat)
print("Mean Scores Carat: " + str(mean_carat))

# Lin Regression auf Basis Laenge, Breite, Hoehe
scores_LBH = cross_val_score(LinearRegression(), X_LBH, Y, cv = RepeatedKFold(n_repeats = 1000))
# print("Scores Carat : " + str(scores_LBH))
mean_LBH = np.mean(scores_LBH)
print("Mean Scores LBH: " + str(mean_LBH))