# inference.py
import joblib
import os
import numpy as np

def model_fn(model_dir):
    """Load the model from the directory"""
    model = joblib.load(os.path.join(model_dir, "random_forest_model.pkl"))
    return model

def input_fn(request_body, request_content_type):
    """Deserialize the request body into an input that the model can predict on"""
    if request_content_type == "text/csv":
        return np.array(request_body.split(",")).reshape(1, -1)
    else:
        raise ValueError("Content type {} is not supported.".format(request_content_type))

def predict_fn(input_data, model):
    """Make predictions"""
    prediction = model.predict(input_data)
    return prediction

def output_fn(prediction, response_content_type):
    """Serialize the prediction into a response that SageMaker can return to the client"""
    return str(prediction[0])
