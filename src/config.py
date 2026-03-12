import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent[1]
FILE_NAME = "sleep_mobile_stress_dataset_15000"
#Paths

DATA_PATH = ROOT_DIR / "data" / "raw" / f"{FILE_NAME}.csv"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"

MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

TEST_SIZE=0.2
RANDOM_STATE=21
SHUFFLE = True
TARGET_COLUMN = 'stress_level'
ID_COLUMN= 'user_id'

NUMERIC_FEATURES = [
    'age', 'daily_screen_time_hours', 'phone_usage_before_sleep_minutes', 
    'sleep_duration_hours', 'sleep_quality_score', 'caffeine_intake_cups', 
    'physical_activity_minutes', 'notifications_received_per_day', 'mental_fatigue_score'
]

CATEGORICAL_FEATURES = []