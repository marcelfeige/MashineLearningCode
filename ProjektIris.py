import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv(
    "D:\Eigene Datein\eLearning Kurse\Machine Learning\Kursmaterialien\Abschnitt 25 - Iris Dataset\iris.csv"
)

print(df.head())

X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values
Y = df["Species"].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 7, test_size = 0.25)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

modelLog = LogisticRegression()
modelLog.fit(X_train, y_train)
scoreLog = modelLog.score(X_test, y_test)
print("score Log: " + str(scoreLog))

modelKNN = KNeighborsClassifier(n_neighbors = 7)
modelKNN.fit(X_train, y_train)
scoreKNN = modelKNN.score(X_test, y_test)
print("score KNN: " + str(scoreKNN))