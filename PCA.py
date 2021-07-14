import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv("..\Kursmaterialien\Abschnitt 37 - PCA\\train.csv.bz2")

print(df.head)

X = df.drop("subject", axis = 1).drop("Activity", axis = 1)
y = df["Activity"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

p = PCA(n_components = 2)
p.fit(X)
X_transformed = p.transform(X)

plt.figure(figsize = (10, 6))

for activity in y.unique():
    X_transformed_filterd = X_transformed[y == activity, :]
    plt.scatter(X_transformed_filterd[:,0], X_transformed_filterd[:, 1], label = activity, s = 1.5)

plt.legend()
plt.show()
