import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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

plt.scatter(X_train[:, 0], X_train[:, 1], c = y_train)
plt.xlabel("Alter")
plt.ylabel("Interesse")
plt.show()

model = LogisticRegression()
model.fit(X_train, y_train)

y_predicted = model.predict(X_test)
score = model.score(X_test, y_test)
print("Score: " + str(score))
