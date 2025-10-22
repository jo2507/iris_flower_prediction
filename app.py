from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

app = Flask(__name__)

# If your saved models are named knn_model.pkl and nb_model.pkl, we will try to load them.
# If they don't exist, we will train quick models and save them automatically.
import os

if os.path.exists('knn_model.pkl'):
    knn_model = joblib.load('knn_model.pkl')
else:
    # quick create and save
    iris = load_iris()
    X, y = iris.data, iris.target
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X, y)
    joblib.dump(knn_model, 'knn_model.pkl')

if os.path.exists('nb_model.pkl'):
    nb_model = joblib.load('nb_model.pkl')
else:
    iris = load_iris()
    X, y = iris.data, iris.target
    nb_model = GaussianNB()
    nb_model.fit(X, y)
    joblib.dump(nb_model, 'nb_model.pkl')

# compute simple accuracy/confusion once and show on page
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

y_pred_knn = knn_model.predict(X_test)
y_pred_nb  = nb_model.predict(X_test)

knn_acc = round(accuracy_score(y_test, y_pred_knn), 3)
nb_acc  = round(accuracy_score(y_test, y_pred_nb), 3)

knn_conf = confusion_matrix(y_test, y_pred_knn)
nb_conf  = confusion_matrix(y_test, y_pred_nb)

def format_conf_matrix(cm):
    return "\n".join(["\t".join(map(str,row)) for row in cm])

@app.route('/', methods=['GET'])
def home():
    # Show default KNN accuracy and confusion on page
    return render_template('index.html',
                           accuracy=f"KNN: {knn_acc}, NB: {nb_acc}",
                           confusion=f"KNN:\n{format_conf_matrix(knn_conf)}\n\nNB:\n{format_conf_matrix(nb_conf)}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sl = float(request.form['sepal_length'])
        sw = float(request.form['sepal_width'])
        pl = float(request.form['petal_length'])
        pw = float(request.form['petal_width'])
        model_sel = request.form.get('model','knn').lower()

        features = np.array([[sl, sw, pl, pw]])
        if model_sel == 'nb':
            pred = nb_model.predict(features)[0]
        else:
            pred = knn_model.predict(features)[0]

        names = ['Setosa','Versicolor','Virginica']
        result = names[int(pred)]

        return render_template('index.html',
                               result=result,
                               accuracy=f"KNN: {knn_acc}, NB: {nb_acc}",
                               confusion=f"KNN:\n{format_conf_matrix(knn_conf)}\n\nNB:\n{format_conf_matrix(nb_conf)}")
    except Exception as e:
        return render_template('index.html', result=f"Error: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
