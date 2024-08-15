# Import necessary modules
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor  # CatBoost regression model
from sklearn.ensemble import (
    AdaBoostRegressor,  # AdaBoost regression model
    GradientBoostingRegressor,  # Gradient Boosting regression model
    RandomForestRegressor,  # Random Forest regression model
)
from sklearn.linear_model import LinearRegression  # Linear Regression model
from sklearn.metrics import r2_score  # Metric for evaluating model performance
from sklearn.neighbors import KNeighborsRegressor  # K-Nearest Neighbors regression model
from sklearn.tree import DecisionTreeRegressor  # Decision Tree regression model
from xgboost import XGBRegressor  # XGBoost regression model

from src.exception import CustomException  # Custom exception handling
from src.logger import logging  # Custom logging setup

from src.utils import save_object, evaluate_models  # Utility functions for saving objects and evaluating models

# Define a data class for model trainer configuration
@dataclass
class ModelTrainerConfig:
    # Define the file path for saving the trained model
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

# Class for handling model training and selection
class ModelTrainer:
    def __init__(self):
        # Initialize the class with the model trainer configuration
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        '''
        This function performs model training, evaluation, and selection
        '''
        try:
            logging.info("Split training and test input data")
            # Split the training and testing data into features and target variables
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # Features for training
                train_array[:, -1],   # Target variable for training
                test_array[:, :-1],   # Features for testing
                test_array[:, -1]     # Target variable for testing
            )

            # Define models to be evaluated
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Define hyperparameters for each model
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # Evaluate models and get the performance report
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

            # Get the best model score from the report
            best_model_score = max(sorted(model_report.values()))

            # Get the best model name from the report
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            # Raise an exception if no model performs well
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            # Save the best model to a file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predict and evaluate the best model
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            # Raise a custom exception if an error occurs
            raise CustomException(e, sys)
