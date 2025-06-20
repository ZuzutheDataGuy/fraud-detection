import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_numerical_features(self, df):
        # Identify numerical features
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        irrelevant_columns = ['application_id', 'customer_id', 'transaction_date', 'transaction_notes', 'fraud_flag_y']
        return [col for col in numerical_features if col not in irrelevant_columns]

    def get_categorical_features(self, df):
        # Identify categorical features
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        irrelevant_columns = ['application_id', 'customer_id', 'transaction_date', 'transaction_notes']
        return [col for col in categorical_features if col not in irrelevant_columns]

    def get_data_transformer_object(self, numerical_columns, categorical_columns):
        try:
            logging.info("Creating preprocessing pipelines for numerical and categorical features.")

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            logging.info("Preprocessing pipelines created successfully.")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def perform_outlier_treatment(self, df, numerical_columns):
        logging.info("Performing outlier treatment on numerical features.")
        for col in numerical_columns:
            lower_bound = df[col].quantile(0.01)
            upper_bound = df[col].quantile(0.99)
            df[col] = np.clip(df[col], lower_bound, upper_bound)
        return df

    def add_aggregated_transaction_features(self, df):
        logging.info("Adding aggregated transaction features.")
        aggregated = df.groupby('customer_id').agg({
            'transaction_amount': ['sum', 'mean', 'max', 'min'],
            'fraud_flag_y': ['sum']
        }).reset_index()

        aggregated.columns = [
            'customer_id', 
            'total_transaction_amount', 
            'avg_transaction_amount', 
            'max_transaction_amount', 
            'min_transaction_amount', 
            'total_fraud_transactions'
        ]

        df = pd.merge(df, aggregated, on='customer_id', how='left')
        return df

    def perform_feature_engineering(self, df):
        logging.info("Performing feature engineering for transactions.")
        df['transaction_speed'] = df['transaction_amount'] / (df['account_balance_after_transaction'] + 1e-6)
        df['income_to_loan_ratio'] = df['monthly_income'] / (df['loan_amount_requested'] + 1e-6)
        return df

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data completed")

            logging.info("Identifying numerical and categorical features.")
            numerical_columns = self.get_numerical_features(train_df)
            categorical_columns = self.get_categorical_features(train_df)

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            # Perform outlier treatment
            train_df = self.perform_outlier_treatment(train_df, numerical_columns)
            test_df = self.perform_outlier_treatment(test_df, numerical_columns)

            # Add aggregated transaction features
            train_df = self.add_aggregated_transaction_features(train_df)
            test_df = self.add_aggregated_transaction_features(test_df)

            # Perform feature engineering
            train_df = self.perform_feature_engineering(train_df)
            test_df = self.perform_feature_engineering(test_df)

            logging.info("Feature engineering and aggregation completed.")

            # Drop target column to create feature sets
            target_column_name = "fraud_flag_y"
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Creating preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object(numerical_columns, categorical_columns)

            logging.info("Applying preprocessing object on train and test data.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info(f"Shape of input_feature_train_arr after transformation: {input_feature_train_arr.shape}")
            logging.info(f"Shape of input_feature_test_arr after transformation: {input_feature_test_arr.shape}")

            # Ensure target variable is reshaped correctly
            target_feature_train_arr = np.array(target_feature_train_df).reshape(-1, 1)
            target_feature_test_arr = np.array(target_feature_test_df).reshape(-1, 1)

            # Concatenate features and target
            train_arr = np.hstack([input_feature_train_arr, target_feature_train_arr])
            test_arr = np.hstack([input_feature_test_arr, target_feature_test_arr])

            logging.info("Preprocessing completed, saving preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)



    