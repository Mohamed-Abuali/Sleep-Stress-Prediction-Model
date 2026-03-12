import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.config import NUMERIC_FEATURES,CATEGORICAL_FEATURES


def get_preprocessing_pipeline():



    preprocessing = ColumnTransformer(
        transformers=[
            ('num',StandardScaler(),NUMERIC_FEATURES),
            ('cat',OneHotEncoder(handle_unknown='ignore'),CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessing

