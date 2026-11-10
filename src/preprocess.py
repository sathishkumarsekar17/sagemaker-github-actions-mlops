import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
train, test = train_test_split(df, test_size=0.2, stratify=df["target"], random_state=42)

for name, frame in [("train", train), ("test", test)]:
    folder = f"/opt/ml/processing/{name}"
    os.makedirs(folder, exist_ok=True)
    frame.to_csv(f"{folder}/{name}.csv", index=False)
print("Prepared:", len(train), len(test))