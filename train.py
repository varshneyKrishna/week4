import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

data = pd.read_csv("iris.csv")

X = data.drop("species", axis=1)
y = data["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")

with open("metrics.txt", "w") as f:
    f.write(f"accuracy: {acc}\n")

print(f"Model trained. Accuracy: {acc}")
