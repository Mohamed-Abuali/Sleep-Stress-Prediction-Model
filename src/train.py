import joblib
import logging
from src.preprocessing import get_model_pipeline,get_preprocessing_pipeline
from src.config import TEST_SIZE,RANDOM_STATE,MODEL_DIR,SHUFFLE
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def train(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=TEST_SIZE,random_state=RANDOM_STATE,shuffle=SHUFFLE)

    preprocessing = get_preprocessing_pipeline()
    pipeline = get_model_pipeline(preprocessing=preprocessing)


    logger.info("Training Model...")
    pipeline.fit(X_train,y_train)

    model_path = MODEL_DIR / "stress_model.joblib"
    joblib.dump(pipeline,model_path)
    logger.info(f"model saved {model_path}")

    return pipeline, X_test,y_test
 
