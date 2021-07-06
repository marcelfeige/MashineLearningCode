import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv(
    "D:\Eigene Datein\eLearning Kurse\Machine Learning"
    "\Kursmaterialien\Abschnitt 05 - Lineare Regression\wohnungspreise.csv")


plt.scatter(df["Quadratmeter"], df["Verkaufspreis"])
plt.show()

model = LinearRegression()
model.fit(df[["Quadratmeter"]], df[["Verkaufspreis"]])

# Verkaufspreis = Intercept + Coef * x
print("Intercept: " + str(model.intercept_))
print("Coef: " + str(model.coef_))

