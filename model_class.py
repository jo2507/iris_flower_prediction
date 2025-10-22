import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle, json

# Load dataset
from sklearn.datasets import load_iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Train Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

# Evaluate
y_pred_knn = knn.predict(X_test)
y_pred_nb = nb.predict(X_test)

print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nConfusion Matrix (KNN):\n", confusion_matrix(y_test, y_pred_knn))
print("\nClassification Report (NB):\n", classification_report(y_test, y_pred_nb))

# Save to Pickle
pickle.dump(knn, open('kneighborsclassifier_model.pkl', 'wb'))
pickle.dump(nb, open('gaussiannb_model.pkl', 'wb'))

# Save to JSON
model_info = {
    "KNN_accuracy": accuracy_score(y_test, y_pred_knn),
    "NB_accuracy": accuracy_score(y_test, y_pred_nb),
    "KNN_report": classification_report(y_test, y_pred_knn, output_dict=True),
    "NB_report": classification_report(y_test, y_pred_nb, output_dict=True)
}

with open("model_info.json", "w") as f:
    json.dump(model_info, f)

print("✅ Models and JSON file saved successfully!")