import pandas as pd
from sklearn.model_selection import train_test_split
from helper import plot_classifier
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv(
    "..\Kursmaterialien\Abschnitt 26 - Entscheidungsbaeume\classification.csv")

df.head()

X = df[["age", "interest"]].values

y = df["success"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.25)

model = GaussianNB()
model.fit(X_test, y_test)

score = model.score(X_test, y_test)
print("R^2: " + str(score))

# Trainings-Daten plotten
plot_classifier(model, X_train, y_train, proba = False, xlabel = "Alter", ylabel = "Interesse")

# Testdaten plotten

plot_classifier(model, X_test, y_test, proba = False, xlabel = "Alter", ylabel = "Interesse")
