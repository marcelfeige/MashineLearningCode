import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from helper import plot_classifier

df = pd.read_csv(
    "..\Kursmaterialien\Abschnitt 21 - logitische regression\classification.csv")

print(df.head())

X = df[["age", "interest"]].values
Y = df["success"].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 0, test_size = 0.25)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors = 7)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print("score: " + str(score))

plot_classifier(model, X_train, y_train, proba = False, xlabel = "Alter", ylabel = "Interesse")