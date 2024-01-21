# Phish-Scanner
Email phishing scanner using svm

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
data = pd.read_csv('emails.csv')
X_train, X_test, y_train, y_test = train_test_split(data.email, data.label, test_size=0.2)
cv = CountVectorizer()
features = cv.fit_transform(X_train)
model = svm.SVC()
model.fit(features, y_train)
features_test = cv.transform(X_test)
print(model.predict(features_test))
