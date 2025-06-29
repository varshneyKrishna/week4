import unittest
import joblib
import pandas as pd

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = joblib.load("models/model.joblib")
        self.sample = pd.DataFrame({
            "sepal_length": [5.1],
            "sepal_width": [3.5],
            "petal_length": [1.4],
            "petal_width": [0.2]
        })

    def test_prediction(self):
        prediction = self.model.predict(self.sample)
        self.assertIn(prediction[0], ['setosa', 'versicolor', 'virginica'])

if __name__ == "__main__":
    unittest.main()
