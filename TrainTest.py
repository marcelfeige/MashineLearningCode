import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv(
    "D:\Eigene Datein\eLearning Kurse\Machine Learning"
    "\Kursmaterialien\Abschnitt 05 - Lineare Regression\wohnungspreise.csv")

df.head()

x = df[["Quadratmeter"]].values # numpy array
y = df[["Verkaufspreis"]].values

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size = 0.25) # 25% testdaten



model = LinearRegression()
model.fit(x_train, y_train)

print("Intercept: " + str(model.intercept_))
print("Coef: " + str(model.coef_))

plt.scatter(x_train, y_train)
plt.scatter(x_test, y_test, color = "green")
plt.show()

predicted = model.predict(x_test)
plt.plot(x_test, predicted)
plt.scatter(x_test, y_test, color = "red")
plt.show()