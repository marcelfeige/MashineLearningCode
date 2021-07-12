import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

df = pd.read_csv("..\Kursmaterialien\Abschnitt 27 - Projekt Pilze (Entscheidungsbaum)\mushrooms.csv")

# Wenn du ein paar Spalten vorab aus den Daten entfernen
# df = df.drop("Spaltenname", axis = 1)

# Wenn du eine kategorische Variable in mehrere Spalten umwandeln
# möchtest, kannst du das mit folgendem Code tun:
df = pd.get_dummies(df)

df.head()

# Welche Spalten sollen zur Vorhersage verwendet werden
df = df.drop("class_e", axis = 1)

# Oder: Die Spalte "success" soll nicht zur Vorhersage verwendet werden:
X = df.drop("class_p", axis = 1).values
y = df["class_p"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.25)

model = DecisionTreeClassifier(criterion = "entropy", max_depth = 4, min_samples_leaf = 3)
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

# Trainings-Daten plotten
#plot_classifier(model, X_train, y_train, proba = False, xlabel = "Alter", ylabel = "Interesse")

plt.figure(dpi = 200)

plot_tree(model,
          feature_names = df.drop("class_p", axis = 1).columns.values,
          class_names = ["essbar", "nicht essbar"],
          filled = True)

plt.show()