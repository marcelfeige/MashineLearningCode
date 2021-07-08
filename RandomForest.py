import pandas as pd
from sklearn.model_selection import train_test_split
from helper import plot_classifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(
    "D:\Eigene Datein\eLearning Kurse\Machine Learning\Kursmaterialien\Abschnitt 26 - Entscheidungsbaeume\classification.csv")

df.head()

X = df[["age", "interest"]].values

y = df["success"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.25)

model = RandomForestClassifier(criterion = "entropy")
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

# Trainings-Daten plotten
plot_classifier(model, X_train, y_train, proba = True, xlabel = "Alter", ylabel = "Interesse")

plt.show()