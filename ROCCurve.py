import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from helper import plot_classifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

df = pd.read_csv("..\Kursmaterialien\Abschnitt 26 - Entscheidungsbaeume\classification.csv")

df.head()

# Welche Spalten sollen zur Vorhersage verwendet werden
X = df[["age", "interest"]].values

# Oder: Die Spalte "success" soll nicht zur Vorhersage verwendet werden:
# X = df.drop("success", axis = 1).values

y = df["success"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.25)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

y_test_pred = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)

plt.plot(fpr,tpr)
plt.show()

rocauc_score = roc_auc_score(y_test, y_test_pred)
print(rocauc_score)

# Trainings-Daten plotten
#plot_classifier(model, X_train, y_train, proba = False, xlabel = "Alter", ylabel = "Interesse")

# Testdaten plotten

#plot_classifier(model, X_test, y_test, proba = False, xlabel = "Alter", ylabel = "Interesse")
