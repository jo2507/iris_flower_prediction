import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

joblib.dump(model, 'nb_model.pkl')
print("nb_model.pkl created successfully!")