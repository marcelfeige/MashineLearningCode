import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv(
    "D:\Eigene Datein\eLearning Kurse\Machine Learning"
    "\Kursmaterialien\Abschnitt 05 - Lineare Regression\wohnungspreise.csv")


plt.scatter(df["Quadratmeter"], df["Verkaufspreis"])
#plt.show()

model = LinearRegression()
model.fit(df[["Quadratmeter"]], df[["Verkaufspreis"]])

# Verkaufspreis = Intercept + Coef * x
print("Intercept: " + str(model.intercept_))
print("Coef: " + str(model.coef_))

# Vorhersagen durch das Model

min_x = min(df["Quadratmeter"])
max_x = max(df["Quadratmeter"])

predicted = model.predict([[20], [110]])
plt.plot([min_x, max_x], predicted, color = "red")
plt.show()