import os
import json
import joblib
import numpy as np

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))

def input_fn(request_body, content_type):
    if content_type != "application/json":
        raise ValueError("Expected application/json")
    values = np.asarray(json.loads(request_body), dtype=float)
    return values.reshape(1, -1) if values.ndim == 1 else values

def predict_fn(input_data, model):
    return model.predict(input_data).tolist()

def output_fn(prediction, accept):
    if accept != "application/json":
        raise ValueError("Expected application/json")
    return json.dumps(prediction), "application/json"