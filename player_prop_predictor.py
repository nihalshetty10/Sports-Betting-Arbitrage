import joblib
import pandas as pd

def predict_prop(features, model_path):
    model = joblib.load(model_path)
    X = features  # select relevant columns
    pred = model.predict(X)
    return pred

# Example usage:
# features = ... # DataFrame for today's props
# pred_points = predict_prop(features, 'points_model.pkl')
