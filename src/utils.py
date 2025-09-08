import os
import joblib

#Save object. For example model or scaler
def save_object(obj, filepath):
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(obj, filepath)
    print(f"Saved object to: {filepath}")

#Load object. Model or scaler
def load_object(filepath):
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    obj = joblib.load(filepath)
    print(f"Loaded object from: {filepath}")
    return obj
