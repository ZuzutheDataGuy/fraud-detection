import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def get_model_configurations(self):
        try:
            logging.info("Setting up model configurations")
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, solver='saga'),
                "LightGBM Classifier": LGBMClassifier(verbose=-1),
                "Random Forest Classifier": RandomForestClassifier(),
                "Gradient Boosting Classifier": GradientBoostingClassifier(),
                "XGBoost Classifier": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "CatBoost Classifier": CatBoostClassifier(verbose=0)
            }
            params = {
                "Logistic Regression": {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2']
                },
                "Random Forest Classifier": {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                },
                "Gradient Boosting Classifier": {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                "XGBoost Classifier": {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                "LightGBM Classifier": {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                "K-Nearest Neighbors": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance']
                },
                "Decision Tree Classifier": {
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                },
                "CatBoost Classifier": {
                    'iterations': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'depth': [3, 5, 7]
                }
            }
            logging.info("Model configurations set up successfully")
            return models, params
        except Exception as e:
            raise CustomException(e, sys)

    def split_data(self, X, y, test_size=0.2, random_state=42):
        try:
            logging.info(f"Splitting data with test_size={test_size}")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            logging.info(f"Train set shape: X_train={X_train.shape}, y_train={y_train.shape}")
            logging.info(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys)

    def apply_smote_sampling(self, X_train, y_train, random_state=42):
        try:
            logging.info("Applying SMOTE for class balancing")
            unique, counts = np.unique(y_train, return_counts=True)
            logging.info(f"Class distribution before SMOTE: {dict(zip(unique, counts))}")
            smote = SMOTE(random_state=random_state)
            X_smote, y_smote = smote.fit_resample(X_train, y_train)
            unique, counts = np.unique(y_smote, return_counts=True)
            logging.info(f"Class distribution after SMOTE: {dict(zip(unique, counts))}")
            logging.info(f"SMOTE applied successfully. New shape: {X_smote.shape}")
            return X_smote, y_smote
        except Exception as e:
            raise CustomException(e, sys)

    def save_train_test_data(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("Saving train and test data")
            train_df = pd.DataFrame(X_train)
            train_df['target'] = y_train
            test_df = pd.DataFrame(X_test)
            test_df['target'] = y_test
            train_df.to_csv(self.model_trainer_config.train_data_path, index=False)
            test_df.to_csv(self.model_trainer_config.test_data_path, index=False)
            logging.info("Train and test data saved successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, X_transformed, y):
        try:
            logging.info("Starting model training process")
            X_train, X_test, y_train, y_test = self.split_data(X_transformed, y)
            X_smote, y_smote = self.apply_smote_sampling(X_train, y_train)
            self.save_train_test_data(X_train, X_test, y_train, y_test)
            models, params = self.get_model_configurations()
            model_report = evaluate_models(X_smote, y_smote, X_test, y_test, models, params)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            logging.info(f"Best model: {best_model_name} with ROC-AUC: {best_model_score:.4f}")
            if best_model_score < 0.6:
                logging.warning(f"Best model score {best_model_score:.4f} is below threshold (0.6)")
                raise CustomException("No best model found with acceptable performance", sys)
            best_model = models[best_model_name]
            if best_model_name in params and params[best_model_name]:
                from sklearn.model_selection import GridSearchCV
                logging.info(f"Training best model {best_model_name} with hyperparameter tuning")
                grid = GridSearchCV(best_model, params[best_model_name], cv=3, scoring='roc_auc', n_jobs=-1)
                grid.fit(X_smote, y_smote)
                best_model = grid.best_estimator_
                logging.info(f"Best parameters: {grid.best_params_}")
            else:
                logging.info(f"Training best model {best_model_name} without hyperparameter tuning")
                best_model.fit(X_smote, y_smote)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Model training completed successfully")
            return (
                self.model_trainer_config.trained_model_file_path,
                self.model_trainer_config.train_data_path,
                self.model_trainer_config.test_data_path
            )
        except Exception as e:
            logging.error("An error occurred during model training")
            raise CustomException(e, sys)

