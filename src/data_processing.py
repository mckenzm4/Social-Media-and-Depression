from kaggle.api.kaggle_api_extended import KaggleApi
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.config import RAW_DATA_PATH


def download_dataset(dataset_slug: str, path: str = RAW_DATA_PATH):
    """
    Download and unzip a Kaggle dataset.

    Args:
        dataset_slug (str): Dataset identifier on Kaggle, e.g. 'adilshamim8/social-media-addiction-vs-relationships'
        path (str): Local folder to download and unzip files into.
    """
    os.makedirs(path, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading dataset '{dataset_slug}' to '{path}'...")
    api.dataset_download_files(dataset_slug, path=path, unzip=True)
    print("Download complete.")

def load_data(path):
   return pd.read_csv(path)

def clean_data(df):
    df_len = len(df)
    df = df.dropna()
    df = df.drop_duplicates()
    cleaned_df_len = len(df)
    print(f"Dropped {df_len - cleaned_df_len} duplicate or empty records.")
    return df

def fit_preprocessor(X):
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numeric_cols)
        ],
        remainder='passthrough'
    )

    preprocessor.fit(X)
    return preprocessor

def transform_with_preprocessor(preprocessor, X):
    return preprocessor.transform(X)

if __name__ == "__main__":

    dataset = 'adilshamim8/social-media-addiction-vs-relationships'
    download_dataset(dataset)

    path = './data/raw/Students Social Media Addiction.csv'

    df = load_data(path)
    df = clean_data(df)

    print(df.head(5))
