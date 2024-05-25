import os
from joblib import load

def load_model(model_name):
    model_path = f"models/{model_name}_model.joblib"
    if os.path.exists(model_path):
        return load(model_path)
    else:
        return None
    
def model_predict(model_name, processed_features):
    
    model = load_model(model_name)
    prediction = model.predict([processed_features])

    return prediction