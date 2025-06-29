import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    loan_applications_raw_path: str = os.path.join('artifacts', 'loan_applications.csv')
    transactions_raw_path: str = os.path.join('artifacts', 'transactions.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def perform_outlier_treatment(self, df, numerical_columns):
        try:
            for col in numerical_columns:
                lower_bound = df[col].quantile(0.01)
                upper_bound = df[col].quantile(0.99)
                df[col] = np.clip(df[col], lower_bound, upper_bound)
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def save_raw_data(self, loan_df, transactions_df):
        try:
            os.makedirs(os.path.dirname(self.ingestion_config.loan_applications_raw_path), exist_ok=True)
            loan_df.to_csv(self.ingestion_config.loan_applications_raw_path, index=False, header=True)
            transactions_df.to_csv(self.ingestion_config.transactions_raw_path, index=False, header=True)
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        try:
            loan_applications_path = 'notebook/data/loan_applications.csv'
            transactions_path = 'notebook/data/transactions.csv'
            loan_df = pd.read_csv(loan_applications_path)
            logging.info("Loan applications dataset loaded successfully.")
            transactions_df = pd.read_csv(transactions_path)
            logging.info("Transactions dataset loaded successfully.")
            numerical_columns_loan = loan_df.select_dtypes(include=[np.number]).columns.tolist()
            numerical_columns_transactions = transactions_df.select_dtypes(include=[np.number]).columns.tolist()
            loan_df = self.perform_outlier_treatment(loan_df, numerical_columns_loan)
            transactions_df = self.perform_outlier_treatment(transactions_df, numerical_columns_transactions)
            self.save_raw_data(loan_df, transactions_df)
            logging.info("Data ingestion process completed successfully.")
            return self.ingestion_config.loan_applications_raw_path, self.ingestion_config.transactions_raw_path
        except Exception as e:
            logging.error("An error occurred during data ingestion.")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        logging.info("Starting data ingestion process.")
        ingestion = DataIngestion()
        loan_data_path, transactions_data_path = ingestion.initiate_data_ingestion()

        logging.info("Starting data transformation process.")
        transformation = DataTransformation()
        transformed_data = transformation.initiate_data_transformation(loan_data_path, transactions_data_path)

        X_transformed, y, preprocessor_path = transformed_data  # Unpack the transformed data

        logging.info("Model training process initiated.")
        trainer = ModelTrainer()

        # Pass X_transformed and y separately
        model_path, train_data_path, test_data_path = trainer.initiate_model_trainer(X_transformed=X_transformed, y=y)

        logging.info(f"Model training completed successfully. Model saved at {model_path}")

    except Exception as e:
        logging.error(f"Error occurred during execution: {e}")

        
