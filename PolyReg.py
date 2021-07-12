import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv(
    "..\\Kursmaterialien\\Abschnitt 12 - Polynomialle Regression\\fields.csv")

print(df.head())

X = df[["width", "length"]].values
Y = df[["profit"]].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 0, test_size = 0.25)

model = LinearRegression()
model.fit(X_train, y_train)

print("Lin Reg: " + str(model.score(X_test, y_test)))


pf = PolynomialFeatures(degree = 2, include_bias = False)
pf.fit(X_train)
X_train_transformed = pf.transform(X_train)
X_test_transformed = pf.transform(X_test)

model.fit(X_train_transformed, y_train)

print("Lin Reg transformed: " + str(model.score(X_test_transformed, y_test)))

plt.scatter(df["length"], df["profit"])
plt.scatter(df["width"], df["profit"], color = "red")
plt.show()


