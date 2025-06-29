import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        self.pipeline_name = "Fraud Detection Train Pipeline"
        
    def run_pipeline(self):
        try:
            logging.info(f"Starting {self.pipeline_name}")
            logging.info("="*50)
            logging.info("STEP 1: DATA INGESTION")
            logging.info("="*50)
            ingestion = DataIngestion()
            loan_data_path, transactions_data_path = ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed successfully")
            logging.info(f"Loan data path: {loan_data_path}")
            logging.info(f"Transactions data path: {transactions_data_path}")
            logging.info("="*50)
            logging.info("STEP 2: DATA TRANSFORMATION")
            logging.info("="*50)
            transformation = DataTransformation()
            X_transformed, y, preprocessor_path = transformation.initiate_data_transformation(
                loan_data_path, transactions_data_path
            )
            logging.info(f"Data transformation completed successfully")
            logging.info(f"Transformed data shape: {X_transformed.shape}")
            logging.info(f"Target variable shape: {y.shape}")
            logging.info(f"Preprocessor saved at: {preprocessor_path}")
            logging.info("="*50)
            logging.info("STEP 3: MODEL TRAINING")
            logging.info("="*50)
            trainer = ModelTrainer()
            model_path, train_path, test_path = trainer.initiate_model_trainer(X_transformed, y)
            logging.info(f"Model training completed successfully")
            logging.info(f"Model saved at: {model_path}")
            logging.info(f"Train data saved at: {train_path}")
            logging.info(f"Test data saved at: {test_path}")
            logging.info("="*50)
            logging.info("PIPELINE SUMMARY")
            logging.info("="*50)
            pipeline_results = {
                'pipeline_status': 'SUCCESS',
                'loan_data_path': loan_data_path,
                'transactions_data_path': transactions_data_path,
                'preprocessor_path': preprocessor_path,
                'model_path': model_path,
                'train_data_path': train_path,
                'test_data_path': test_path,
                'data_shape': {
                    'features': X_transformed.shape,
                    'target': y.shape
                }
            }
            logging.info(f"✅ {self.pipeline_name} completed successfully!")
            logging.info(f"📁 Model saved at: {model_path}")
            logging.info(f"🔧 Preprocessor saved at: {preprocessor_path}")
            logging.info(f"📊 Train data saved at: {train_path}")
            logging.info(f"📊 Test data saved at: {test_path}")
            return pipeline_results
        except Exception as e:
            logging.error(f"❌ {self.pipeline_name} failed")
            logging.error(f"Error: {str(e)}")
            pipeline_results = {
                'pipeline_status': 'FAILED',
                'error_message': str(e),
                'failed_step': self.get_current_step(e)
            }
            raise CustomException(e, sys)
    
    def get_current_step(self, error):
        error_str = str(error).lower()
        if 'ingestion' in error_str or 'loading' in error_str:
            return 'Data Ingestion'
        elif 'transformation' in error_str or 'preprocessing' in error_str:
            return 'Data Transformation'
        elif 'training' in error_str or 'model' in error_str:
            return 'Model Training'
        else:
            return 'Unknown'
