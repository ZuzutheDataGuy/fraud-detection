import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    loan_applications_raw_path: str = os.path.join('artifacts', 'loan_applications.csv')
    transactions_raw_path: str = os.path.join('artifacts', 'transactions.csv')
    combined_raw_path: str = os.path.join('artifacts', 'combined_raw.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting the data ingestion process for multiple datasets.")
        try:
            
            loan_applications_path = 'notebook/data/loan_applications.csv'
            transactions_path = 'notebook/data/transactions.csv'
            
            loan_df = pd.read_csv(loan_applications_path)
            logging.info("Loan applications dataset loaded successfully.")
            transactions_df = pd.read_csv(transactions_path)
            logging.info("Transactions dataset loaded successfully.")

            os.makedirs(os.path.dirname(self.ingestion_config.loan_applications_raw_path), exist_ok=True)
            loan_df.to_csv(self.ingestion_config.loan_applications_raw_path, index=False, header=True)
            transactions_df.to_csv(self.ingestion_config.transactions_raw_path, index=False, header=True)
            logging.info("Raw datasets saved to artifacts directory.")

            combined_df = pd.merge(loan_df, transactions_df, on='customer_id', how='left')
            combined_df.to_csv(self.ingestion_config.combined_raw_path, index=False, header=True)
            logging.info("Combined dataset created and saved.")
            
            train_set, test_set = train_test_split(combined_df, test_size=0.2, random_state=42)
            logging.info("Combined dataset split into training and testing sets.")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Training and testing datasets saved to artifacts directory.")

            logging.info("Data ingestion process completed successfully.")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error("An error occurred during data ingestion.")
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)
    
    

