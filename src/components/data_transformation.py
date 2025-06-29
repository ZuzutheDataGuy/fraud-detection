import os
import sys
import pandas as pd
import numpy as np
from datetime import timedelta
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
import pickle


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    transformed_train_data_path: str = os.path.join("artifacts", "transformed_data.csv")
    feature_names_path: str = os.path.join("artifacts", "feature_names.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.epsilon = 1e-6

    def create_datetime_features(self, df):
        try:
            logging.info("Creating datetime features")
            df['application_date'] = pd.to_datetime(df['application_date'])
            df['application_year'] = df['application_date'].dt.year
            df['application_month'] = df['application_date'].dt.month
            df['application_day_of_week'] = df['application_date'].dt.dayofweek
            df['application_hour'] = df['application_date'].dt.hour
            df['is_weekend'] = df['application_date'].dt.weekday > 4
            logging.info("Datetime features created successfully")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def create_loan_features(self, df):
        try:
            logging.info("Creating loan-related features")
            df['loan_affordability_ratio'] = df['loan_amount_requested'] / (df['monthly_income'] + self.epsilon)
            df['existing_emi_to_income_ratio'] = (
                df['existing_emis_monthly'] / (df['monthly_income'] + self.epsilon)
            ) * 100
            logging.info("Loan features created successfully")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def aggregate_transaction_features(self, merged_df):
        try:
            logging.info("Starting transaction feature aggregation")
            time_windows = [30, 90, 180, 365]
            aggregated_transaction_features = []

            merged_df['transaction_date'] = pd.to_datetime(merged_df['transaction_date'])
            merged_df = merged_df.sort_values(by=['customer_id', 'transaction_date'])

            for customer_id, customer_group in merged_df.groupby('customer_id'):
                for _, loan_row in customer_group.drop_duplicates(subset='application_id').iterrows():
                    application_date = loan_row['application_date']
                    application_id = loan_row['application_id']

                    transactions_before_application = customer_group[
                        customer_group['transaction_date'] < application_date
                    ].copy()

                    app_features = {'application_id': application_id}

                    for window_days in time_windows:
                        window_start_date = application_date - timedelta(days=window_days)
                        transactions_in_window = transactions_before_application[
                            transactions_before_application['transaction_date'] >= window_start_date
                        ]

                        app_features[f'num_transactions_{window_days}d'] = transactions_in_window.shape[0]
                        app_features[f'total_transaction_amount_{window_days}d'] = transactions_in_window['transaction_amount'].sum()
                        app_features[f'average_transaction_amount_{window_days}d'] = transactions_in_window['transaction_amount'].mean() if transactions_in_window.shape[0] > 0 else 0
                        app_features[f'unique_merchant_categories_{window_days}d'] = transactions_in_window['merchant_category'].nunique()

                    aggregated_transaction_features.append(app_features)

            transaction_aggregation_df = pd.DataFrame(aggregated_transaction_features)
            logging.info("Transaction feature aggregation completed successfully")
            return transaction_aggregation_df
        except Exception as e:
            raise CustomException(e, sys)

    def aggregate_fraud_features(self, merged_df):
        try:
            logging.info("Starting fraud feature aggregation")
            aggregated_fraud_features = []

            for customer_id, group in merged_df.groupby('customer_id'):
                for _, loan_row in group.drop_duplicates(subset='application_id').iterrows():
                    application_date = loan_row['application_date']
                    prior_transactions = group[group['transaction_date'] < application_date]

                    fraud_metrics = {
                        'application_id': loan_row['application_id'],
                        'failed_transactions_count': (prior_transactions['transaction_status'] == 'Failed').sum(),
                        'international_transactions_count': prior_transactions['is_international_transaction'].sum(),
                    }

                    aggregated_fraud_features.append(fraud_metrics)

            fraud_features_df = pd.DataFrame(aggregated_fraud_features)
            logging.info("Fraud feature aggregation completed successfully")
            return fraud_features_df
        except Exception as e:
            raise CustomException(e, sys)

    def merge_datasets(self, loan_applications_path, transactions_path):
        try:
            logging.info("Loading datasets for transformation")
            loan_applications_df = pd.read_csv(loan_applications_path)
            loan_applications_df['application_date'] = pd.to_datetime(loan_applications_df['application_date'])
            transactions_df = pd.read_csv(transactions_path)
            transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])

            logging.info("Merging datasets")
            merged_df = pd.merge(
                loan_applications_df,
                transactions_df,
                on='customer_id',
                how='left'
            )
            logging.info("Datasets merged successfully")
            return loan_applications_df, transactions_df, merged_df
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self, X):
        try:
            logging.info("Creating data transformer object")
            
            numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if 'residential_address' in categorical_features:
                categorical_features.remove('residential_address')
                logging.info("Removed 'residential_address' from categorical features")

            logging.info(f"Numerical features: {numerical_features}")
            logging.info(f"Categorical features: {categorical_features}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ], remainder='drop'
            )

            logging.info("Data transformer object created successfully")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def save_object(self, file_path, obj):
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, loan_applications_path, transactions_path):
        try:
            logging.info("Starting data transformation process")

            loan_applications_df, transactions_df, merged_df = self.merge_datasets(
                loan_applications_path, transactions_path
            )

            loan_applications_df = self.create_datetime_features(loan_applications_df)

            loan_applications_df = self.create_loan_features(loan_applications_df)

            transaction_aggregation_df = self.aggregate_transaction_features(merged_df)

            fraud_features_df = self.aggregate_fraud_features(merged_df)

            logging.info("Merging aggregated features")
            loan_applications_df = pd.merge(
                loan_applications_df, transaction_aggregation_df, 
                on='application_id', how='left'
            )
            loan_applications_df = pd.merge(
                loan_applications_df, fraud_features_df, 
                on='application_id', how='left'
            )

            logging.info("Preparing features and target variable")
            y = loan_applications_df['fraud_flag']
            X = loan_applications_df.drop(columns=[
                'fraud_flag', 'fraud_type', 'loan_status', 
                'application_id', 'customer_id', 'application_date'
            ])

            X = X.fillna(0)

            preprocessing_obj = self.get_data_transformer_object(X)

            logging.info("Applying transformations")
            X_transformed = preprocessing_obj.fit_transform(X)

            if hasattr(X_transformed, 'toarray'):
                X_transformed = X_transformed.toarray()

            logging.info(f"Shape of transformed data: {X_transformed.shape}")
            logging.info(f"Shape of target variable: {y.shape}")

            logging.info("Saving preprocessing object")
            self.save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            try:
                feature_names = preprocessing_obj.get_feature_names_out()
                logging.info("Successfully retrieved feature names from preprocessor")
            except Exception as e:
                logging.warning(f"Could not get feature names from preprocessor: {e}")
                numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
                
                if 'residential_address' in categorical_features:
                    categorical_features.remove('residential_address')
                
                feature_names = []
                feature_names.extend([f"num__{name}" for name in numerical_features])
                
                try:
                    cat_transformer = preprocessing_obj.named_transformers_['cat']
                    for i, feature in enumerate(categorical_features):
                        categories = cat_transformer.categories_[i]
                        feature_names.extend([f"cat__{feature}__{cat}" for cat in categories])
                except:
                    feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
                    
            logging.info(f"Number of feature names: {len(feature_names)}")
            logging.info(f"First 10 feature names: {feature_names[:10]}")
            
            self.save_object(
                file_path=self.data_transformation_config.feature_names_path,
                obj=feature_names
            )

            logging.info("Creating transformed dataset")
            
            target_array = np.array(y).flatten()
            
            transformed_data = pd.DataFrame(X_transformed, columns=feature_names)
            transformed_data['target'] = target_array
            
            logging.info(f"Final transformed data shape: {transformed_data.shape}")
            
            transformed_data.to_csv(
                self.data_transformation_config.transformed_train_data_path, 
                index=False
            )

            logging.info("Data transformation completed successfully")

            return (
                X_transformed,
                target_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error("An error occurred during data transformation")
            raise CustomException(e, sys)
