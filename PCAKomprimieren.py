import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("..\Kursmaterialien\Abschnitt 37 - PCA\\train.csv.bz2")
test = pd.read_csv("..\Kursmaterialien\Abschnitt 37 - PCA\\test.csv.bz2")


X_train = train.drop("subject", axis = 1).drop("Activity", axis = 1)
y_train = train["Activity"]

X_test = test.drop("subject", axis = 1).drop("Activity", axis = 1)
y_test = test["Activity"]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

p = PCA(n_components = 5)

X_train_transformed = p.fit_transform(X_train)
X_test_trandformed = p.transform(X_test)

clf = LogisticRegression(max_iter=10000)
clf.fit(X_train_transformed, y_train)
score = clf.score(X_test_trandformed,y_test)
print(score)


