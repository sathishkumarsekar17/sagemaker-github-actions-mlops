import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv("/opt/ml/input/data/train/train.csv")
test = pd.read_csv("/opt/ml/input/data/test/test.csv")
X_train, y_train = train.drop(columns="target"), train["target"]
X_test, y_test = test.drop(columns="target"), test["target"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("test_accuracy:", accuracy_score(y_test, model.predict(X_test)))
os.makedirs("/opt/ml/model", exist_ok=True)
joblib.dump(model, "/opt/ml/model/model.joblib")