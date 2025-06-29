import os
import sys
import dill

from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score


def save_object(file_path: str, obj: object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    results = {}
    for model_name, model in models.items():
        try:
            logging.info(f"Evaluating model: {model_name}")
            
            if model_name in params and params[model_name]:
            
                from sklearn.model_selection import GridSearchCV
                grid = GridSearchCV(model, params[model_name], cv=3, scoring='roc_auc', n_jobs=-1)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
            else:
               
                model.fit(X_train, y_train)
                best_model = model

            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            results[model_name] = roc_auc

        except Exception as e:
            logging.error(f"Error evaluating model {model_name}: {e}")
            results[model_name] = None
    
    return results

def load_object(file_path: str):
    try:
        with open(file_path, 'rb') as file_obj:
            obj = dill.load(file_obj)
        logging.info(f"Object loaded from {file_path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys)
