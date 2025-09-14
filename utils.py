"""utils.py
Helper utilities: model save/load, evaluation metrics and simple plotting.
"""
import joblib
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def evaluate_regression(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    print(f'RMSE: {rmse:.4f} | R2: {r2:.4f}')
    return {'rmse': rmse, 'r2': r2}

def postprocess_predictions(preds):
    # placeholder for any postprocessing (clipping, inverse transforms, etc.)
    return preds
