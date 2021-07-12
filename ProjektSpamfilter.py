import pandas as pd
from sklearn.model_selection import train_test_split
from helper import plot_classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("..\Kursmaterialien\Abschnitt 31 - Projekt Spam-Filter\spam.csv")

print(df.head())

X = df["message"]

y = df["type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

model = MultinomialNB()
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print("Score: " + str(score))