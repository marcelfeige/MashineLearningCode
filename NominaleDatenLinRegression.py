import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("D:\Eigene Datein\eLearning Kurse\Machine Learning\Kursmaterialien"
                 "\Abschnitt 08 - Lineare Regression mit mehreren Variablen\hotels.csv")


df = pd.get_dummies(df, columns = ["Stadt"])

print(df.head())

X = df [["Gewinn", "Quadratmeter", "Stadt_Berlin", "Stadt_Köln"]].values
Y = df[["Preis in Mio"]].values

X_test, X_train, y_test, y_train = train_test_split(X, Y, random_state = 0, test_size = 0.25)

model = LinearRegression()
model.fit(X_test, y_test)

r2modelScore = model.score(X_test, y_test)
print("R^2: " + str(r2modelScore))

