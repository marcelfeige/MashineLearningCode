import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("D:\Eigene Datein\eLearning Kurse\Machine Learning\Kursmaterialien"
                 "\Abschnitt 08 - Lineare Regression mit mehreren Variablen\hotels.csv")

print(df.head())

X = df[["Gewinn", "Quadratmeter"]].values
Y = df[["Preis in Mio"]].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 0, test_size = 0.25)

model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept: " + str(model.intercept_))
print("Coef: " + str(model.coef_))

y_test_pred = model.predict(X_test)

for i in range (0, len(y_test_pred)):
    print(str(y_test_pred[i][0]) + " - " + str(y_test[i][0]))




