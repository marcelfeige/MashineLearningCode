import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("..\Kursmaterialien\Abschnitt 13 - Praxis Polynomiale Regression\diamonds.csv")

print(df.head())

# Preis auf Basis des Gewichts schaetzen (Lineare Regression)
X_carat = df[["carat"]].values
Y_price = df[["price"]].values

X_carat_train, X_carat_test, y_price_train, y_price_test = train_test_split(X_carat, Y_price, random_state = 0, test_size = 0.25)

model = LinearRegression()
model.fit(X_carat_train, y_price_train)

score_carat_price = model.score(X_carat_test, y_price_test)
print("Score Lin carat - price: " + str(score_carat_price))

# Preis auf Basis der Laenge, Breite und Hoehe schaetzen (Lineare Regression)
X_LBH = df[["x", "y", "z"]]

X_LBH_train, X_LBH_test, y_price_train, y_price_test = train_test_split(X_LBH , Y_price, random_state = 0, test_size = 0.25)

model.fit(X_LBH_train, y_price_train)

score_LBH_price_lin = model.score(X_LBH_test, y_price_test)
print("Score Lin LBH - price: " + str(score_LBH_price_lin))

# Preis auf Basis der Laenge, Breite und Hoehe schaetzen (Polynomiale Regression)
pf = PolynomialFeatures(degree = 2, include_bias = False)
pf.fit(X_LBH_train)

X_LBH_train_transformed = pf.transform(X_LBH_train)
X_LBH_test_transformed = pf.transform(X_LBH_test)

model.fit(X_LBH_train_transformed, y_price_train)
score_LBH_price_poly = model.score(X_LBH_test_transformed, y_price_test)
print("Score Poly LBH - price: " + str(score_LBH_price_poly))
