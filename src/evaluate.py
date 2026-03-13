import logging
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


logger = logging.getLogger(__name__)

def evaluate_model(model,X_test,y_test):
    y_pred = model.predict(X_test)

    metrics = {
        'r2':r2_score(y_test,y_pred),
        'mse':mean_squared_error(y_test,y_pred),
        'mae':mean_absolute_error(y_test,y_pred)
    }

    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")

    return metrics
