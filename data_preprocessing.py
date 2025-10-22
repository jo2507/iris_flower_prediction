import pandas as pd
from sklearn.datasets import load_iris

class DataPreprocessor:
    def _init_(self):
        self.data = None
        self.df = None

    def load_data(self):
        iris = load_iris()
        self.data = iris
        self.df = pd.DataFrame(
            iris.data, columns=iris.feature_names
        )
        self.df['target'] = iris.target
        print("✅ Data loaded successfully!")
        print(self.df.head())
        return self.df

if __name__ == "__main__":
    dp = DataPreprocessor()
    dp.load_data()