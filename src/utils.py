import os
import sys
import dill
from src.logger import logging
from src.exception import CustomException

def save_object(file_path: str, obj: object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)
