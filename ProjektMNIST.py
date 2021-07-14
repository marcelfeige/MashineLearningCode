import gzip
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

train_data = os.path.join("..", "Kursmaterialien", "data", "mnist", "train-images-idx3-ubyte.gz")
train_labels = os.path.join("..", "Kursmaterialien", "data", "mnist", "train-labels-idx1-ubyte.gz")

test_data = os.path.join("..", "Kursmaterialien", "data", "mnist", "t10k-images-idx3-ubyte.gz")
test_labels = os.path.join("..", "Kursmaterialien", "data", "mnist", "t10k-labels-idx1-ubyte.gz")

def mnist_images(filename):
    with gzip.open(filename, "rb") as file:
        # offset: erste 16 eintraege sind meta daten
        data = np.frombuffer(file.read(), np.uint8, offset = 16)
        # Pixelgroesse fuer jedes Bild 28x28
        # /255: umwandlung in kommazahlen fuer bessere berechnungen
        return data.reshape(-1, 28, 28)/255

    print(filename)

def mnist_labels(filename):
    with gzip.open(filename, "rb") as file:
        # offset: erste 8 eintraege sind meta daten
        # unit8: zahlen zwischen 0 und 255
        return np.frombuffer(file.read(), np.uint8, offset = 8)


X_train = mnist_images(train_data)
y_train = mnist_labels(train_labels)

X_test = mnist_images(test_data)
y_test = mnist_labels(test_labels)

# print(y_train[2])
# plt.imshow(X_train[2])
# plt.show()

#caler = StandardScaler()
#scaler.fit(X_train)

modellOG = LogisticRegression(max_iter = 10000)
# reshape: alle eintrage stehen hintereinander und nicht zeile für zeile in arrays
modellOG.fit(X_train.reshape(-1, 784), y_train)
scoreLog = modellOG.score(X_test.reshape(-1, 784), y_test)
print("Score Log: " + str(scoreLog))

modelLin = LinearRegression()
modelLin.fit(X_train.reshape(-1, 784), y_train)
scoreLin = modellOG.score(X_test.reshape(-1, 784), y_test)
print("Score Lin: " +str(scoreLin))