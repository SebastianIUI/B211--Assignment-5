import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model1 = LogisticRegression(max_iter=10000)
model1.fit(X_train, y_train)
pred1 = model1.predict(X_test)

model2 = DecisionTreeClassifier()
model2.fit(X_train, y_train)
pred2 = model2.predict(X_test)

model3 = KNeighborsClassifier()
model3.fit(X_train, y_train)
pred3 = model3.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, pred1))
print("Decision Tree Accuracy:", accuracy_score(y_test, pred2))
print("KNN Accuracy:", accuracy_score(y_test, pred3))

