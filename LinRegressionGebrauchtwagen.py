import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv(
    "D:\Eigene Datein\\eLearning Kurse\\Machine Learning\Kursmaterialien\\"
    "Abschnitt 06 - Projekt Lineare Regression\\autos_prepared.csv")


plt.scatter(df["kilometer"], df["price"])
#plt.show()

model = LinearRegression()
model.fit(df[["kilometer"]], df[["price"]])

# Verkaufspreis = Intercept + Coef * x
print("Intercept: " + str(model.intercept_))
print("Coef: " + str(model.coef_))

# Vorhersagen durch das Model

min_x = min(df["kilometer"])
max_x = max(df["kilometer"])

p50k = model.predict([[50000]])
print("Preis bei 50.000 km: " + str(p50k))

predicted = model.predict([[min_x], [max_x]])
plt.plot([min_x, max_x], predicted, color = "red")
plt.show()
