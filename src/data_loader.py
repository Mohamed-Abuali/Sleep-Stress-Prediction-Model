import os
import logging
import pandas as pd
from src.config import DATA_PATH,TARGET_COLUMN,ID_COLUMN
logger = logging.getLogger(__name__)
def load_data() -> pd.DataFrame:
    try:
        data = pd.read_csv(DATA_PATH)
        logger.info("data was successfully loaded",data.head())

        if TARGET_COLUMN not in data.columns:
            raise ValueError(f"Target column: {TARGET_COLUMN} was not found")
        return data
    except Exception as e:
        print(f"An error accurd while load the data {e}")


def prepare_feature(df:pd.DataFrame) -> tuple:

    X = df.drop(columns=[TARGET_COLUMN,ID_COLUMN], errors='ignore')
    y = df[TARGET_COLUMN]
    return X,y
