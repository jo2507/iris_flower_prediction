from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a simple KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Save the model
joblib.dump(knn, 'knn_model.pkl')

print("✅ knn_model.pkl created successfully!")