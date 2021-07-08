import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv(
    "D:\Eigene Datein\eLearning Kurse\Machine Learning\Kursmaterialien\Abschnitt 21 - logitische regression\classification.csv")

print(df.head())

X = df[["age", "interest"]].values
Y = df["success"].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 0, test_size = 0.25)

plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.xlabel("Alter")
plt.ylabel("Interesse")
plt.show()