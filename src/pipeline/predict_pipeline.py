import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.pipeline_name = "Fraud Detection Predict Pipeline"
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.model_path = os.path.join("artifacts", "model.pkl")

    def load_artifacts(self):
        try:
            logging.info("Loading preprocessor and model artifacts")
            preprocessor = load_object(self.preprocessor_path)
            logging.info("Preprocessor loaded successfully")
            model = load_object(self.model_path)
            logging.info("Model loaded successfully")
            return preprocessor, model
        except Exception as e:
            raise CustomException(e, sys)

    def preprocess_data(self, input_data, preprocessor):
        try:
            logging.info("Preprocessing input data")
            if not isinstance(input_data, pd.DataFrame):
                input_data = pd.DataFrame(input_data)
            transformed_data = preprocessor.transform(input_data)
            logging.info("Input data preprocessed successfully")
            return transformed_data
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, input_data):
        try:
            logging.info(f"Starting {self.pipeline_name}")
            preprocessor, model = self.load_artifacts()
            transformed_data = self.preprocess_data(input_data, preprocessor)
            predictions = model.predict(transformed_data)
            prediction_probs = model.predict_proba(transformed_data)
            logging.info(f"Predictions made successfully: {predictions}")
            return {
                "predictions": predictions,
                "prediction_probabilities": prediction_probs
            }
        except Exception as e:
            logging.error(f"❌ {self.pipeline_name} failed")
            logging.error(f"Error: {str(e)}")
            raise CustomException(e, sys)
