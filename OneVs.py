import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier

df = pd.read_csv(
    "..\Kursmaterialien\Abschnitt 23 - Mehrere Klassen\\foods.csv")
print(df.head())

X = df[["energy_100g", "fat_100g", "carbohydrates_100g", "sugars_100g", "proteins_100g"]].values
Y = df["clss"].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 0, test_size = 0.25)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

modelOVR = LogisticRegression(multi_class = "ovr")
modelOVR.fit(X_train, y_train)

print(modelOVR.score(X_test, y_test))

modelOVO = OneVsOneClassifier(LogisticRegression())
modelOVO.fit(X_train, y_train)
print(modelOVO.score(X_test, y_test))
