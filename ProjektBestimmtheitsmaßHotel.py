import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("..\Kursmaterialien\Abschnitt 08 - Lineare Regression mit mehreren Variablen\hotels.csv")

# print(df.head())

X = df[["Gewinn", "Quadratmeter"]].values
Y = df[["Preis in Mio"]].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 0, test_size = 0.25)

model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept: " + str(model.intercept_))
print("Coef: " + str(model.coef_))

y_test_pred = model.predict(X_test)

r2score = r2_score(y_test, y_test_pred)

print("R^2: " + str(r2score))

# alternativ

r2modelScore = model.score(X_test, y_test)
print("R^2: " + str(r2modelScore))





