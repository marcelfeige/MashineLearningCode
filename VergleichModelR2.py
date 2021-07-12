import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("..\Kursmaterialien\\Abschnitt 06 - Projekt Lineare Regression\\autos_prepared.csv")

scores_km_price = []
scores_ps_price = []
scores_km_ps_price = []
model = LinearRegression()

for i in range(0, 100):

    X = df[["kilometer"]]
    Y = df[["price"]]

    model.fit(X, Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)

    scores_km_price.append(model.score(X_test, y_test))

print("R^2 km - Preis: " + str(sum(scores_km_price) / len(scores_km_price)))

for i in range(0, 100):

    X = df[["powerPS"]]
    Y = df[["price"]]

    model.fit(X, Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)

    scores_ps_price.append(model.score(X_test, y_test))

print("R^2 PS - Preis: " + str(sum(scores_ps_price) / len(scores_ps_price)))

for i in range(0, 100):

    X = df[["kilometer","powerPS"]]
    Y = df[["price"]]

    model.fit(X, Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)

    scores_km_ps_price.append(model.score(X_test, y_test))

print("R^2 km, PS - Preis: " + str(sum(scores_km_ps_price) / len(scores_km_ps_price)))


# # Verkaufspreis = Intercept + Coef * x
# print("Intercept: " + str(model.intercept_))
# print("Coef: " + str(model.coef_))
#
#
# # Vorhersagen durch das Model
# min_x = min(df["kilometer"])
# max_x = max(df["kilometer"])
#
# predicted = model.predict([[min_x], [max_x]])
# plt.scatter(df["kilometer"], df["price"])
# plt.plot([min_x, max_x], predicted, color = "red")
# plt.xlabel("Kilometer")
# plt.ylabel("Preis")
# plt.show()
