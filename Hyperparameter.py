import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from helper import plot_classifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("..\Kursmaterialien\Abschnitt 26 - Entscheidungsbaeume\classification.csv")


# Welche Spalten sollen zur Vorhersage verwendet werden
X = df[["age", "interest"]].values
y = df["success"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
pipeline = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])

pipeline.fit(X_train, y_train)
pip_score = pipeline.score(X_test, y_test)

print(pip_score)

clf = GridSearchCV(pipeline, param_grid = {
    "knn__n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
clf.fit(X, y)

print(clf.best_params_)

score = clf.best_score_

print(score)

# Trainings-Daten plotten
#plot_classifier(model, X_train, y_train, proba = False, xlabel = "Alter", ylabel = "Interesse")

# Testdaten plotten

#plot_classifier(model, X_test, y_test, proba = False, xlabel = "Alter", ylabel = "Interesse")
