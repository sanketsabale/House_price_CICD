import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = ["area","bedrooms","age","price"]

def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found:{path}")
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(f"missing columns: {missing}")
    
    return df

def split_data(df,test_size = 0.2):
    x = df[["area","bedrooms","age"]]
    y = df["price"]
    return train_test_split(x, y, test_size=test_size, random_state=42)



def save_model(model,path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError("model file not found")
    return joblib.load(path)

