"""
Model training component.

- Defines a set of regression models and their hyperparameter grids.
- Uses evaluate_models to train and compare them.
- Selects the best-performing model and saves it to disk.
"""

import os
import sys
from dataclasses import dataclass
from typing import Tuple

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#from sklearn.neighbors import KNeighborsRegressor  #currently unused, but can be added later
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_models


@dataclass
class ModelTrainerConfig:
    """
    Configuration for the model trainer.

    Attributes:
        train_model_file_path: Path where the best trained model will be saved.
    """
    train_model_file_path: str = os.path.join("artifact", "model.pkl")


class ModelTrainer:
    """
    Model training pipeline step.

    Responsible for:
    - Splitting the transformed arrays into X/y train and test.
    - Defining models and hyperparameter grids.
    - Calling evaluate_models to train and compare models.
    - Selecting the best model and saving it.
    """

    def __init__(self) -> None:
        """
        Initialize the ModelTrainer with its configuration.
        """
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array) -> tuple[str, float]:
        """
        Train multiple models, select the best one, and save it.

        Args:
            train_array: Numpy array with transformed training features and target
                         (target is assumed to be the last column).
            test_array: Numpy array with transformed test features and target
                        (target is assumed to be the last column).

        Returns:
            A tuple of:
                (best_model_name, final_r2)
            where final_r2 is the R² score of the best model on the test set.

        Raises:
            CustomException: If no sufficiently good model is found or any
                             error occurs during training.
        """
        try:
            logging.info("Splitting transformed arrays into train/test features and targets.")

            # Split arrays into features (all columns except last) and target (last column)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Dictionary of candidate models
            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(random_state=42),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False, random_state=42),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
            }

            # Hyperparameters for GridSearchCV
            params = {
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                },
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            logging.info("Starting model evaluation using evaluate_models.")

            # model_report: {name: {"train_r2": ..., "test_r2": ...}, ...}
            # fitted_models: {name: best_fitted_model, ...}
            model_report, fitted_models = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
            )

            logging.info(f"Model evaluation report: {model_report}")

            # Find the best model based on test R² score
            best_model_name: str | None = None
            best_model_score = float("-inf")

            for name, scores in model_report.items():
                test_r2 = scores.get("test_r2", float("-inf"))
                if test_r2 > best_model_score:
                    best_model_score = test_r2
                    best_model_name = name

            if best_model_name is None:
                raise CustomException(Exception("No models were evaluated."), sys)

            logging.info(
                f"Best model based on test R²: {best_model_name} "
                f"with score: {best_model_score:.4f}"
            )

            # Enforce a minimum acceptable performance
            if best_model_score < 0.6:
                raise CustomException(
                    Exception("All model scores are below 0.60; no suitable model found."),
                    sys,
                )

            best_model = fitted_models[best_model_name]

            logging.info("Saving best model to artifact directory.")

            # Save the best model to disk
            save_obj(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model,
            )

            # Final evaluation on test set for return
            predicted = best_model.predict(X_test)
            final_r2 = r2_score(y_test, predicted)

            logging.info(f"Final test R² of the saved model: {final_r2:.4f}")

            # ✅ Return BOTH model name and test R²
            return best_model_name, float(final_r2)

        except Exception as e:
            # Wrap any error that occurs in training/selection/saving
            raise CustomException(e, sys)


