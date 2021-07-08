import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from helper import plot_classifier

df = pd.read_csv(
    "D:\Eigene Datein\eLearning Kurse\Machine Learning\Kursmaterialien\Abschnitt 22 - Brustkrebs erkennen\cancer.csv")

# Wenn du ein paar Spalten vorab aus den Daten entfernen
# df = df.drop("Spaltenname", axis = 1)

print(df.head())

df = df.drop("id", axis = 1)

print(df.head())

# Welche Spalten sollen zur Vorhersage verwendet werden
X = df.drop("diagnosis", axis = 1).values
y = df["diagnosis"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.25)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

print("score: " + str(model.score(X_test, y_test)))

