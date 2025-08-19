import os
import joblib

def save_object(obj, filepath):
    """
    Save a Python object to a file using joblib.
    
    Args:
        obj: The Python object to save (e.g., model, preprocessor, matrix).
        filepath: Full path to the output .joblib file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(obj, filepath)
    print(f"Saved object to: {filepath}")

def load_object(filepath):
    """
    Load a Python object from a .joblib file.
    
    Args:
        filepath: Full path to the .joblib file to load.
    
    Returns:
        The loaded Python object.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    obj = joblib.load(filepath)
    print(f"Loaded object from: {filepath}")
    return obj
