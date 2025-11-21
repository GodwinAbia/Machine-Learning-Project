"""
Utility functions for model persistence and evaluation.

- save_obj: save a Python object (e.g. model, preprocessor) to disk using dill.
- load_obj: load a previously saved Python object from disk.
- evaluate_models: train and evaluate multiple models with optional GridSearchCV.
"""

import os
import sys
from typing import Any, Dict, Tuple

import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_obj(file_path: str, obj: Any) -> None:
    """
    Save a Python object to disk using dill.

    Args:
        file_path: Full path (including filename) where the object will be stored.
        obj: Any Python object (e.g. trained model, preprocessor) to serialize.

    Notes:
        - Creates the parent directory if it does not exist.
    """
    try:
        dir_path = os.path.dirname(file_path)

        # Only try to create the directory if there is one in the path
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # "wb" = write binary mode
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        # Wrap any error with our custom exception for better trace info
        raise CustomException(e, sys)


def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models: Dict[str, Any],
    params: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """
    Train and evaluate multiple models with optional hyperparameter tuning.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.
        models: Dict of model name -> model instance, e.g.
            {
                "RandomForest": RandomForestRegressor(),
                "LinearRegression": LinearRegression(),
                ...
            }
        params: Dict of model name -> param grid for GridSearchCV, e.g.
            {
                "RandomForest": {"n_estimators": [50, 100]},
                "LinearRegression": {},
                ...
            }

    Returns:
        report: Dict mapping model_name -> dict of scores, e.g.
            {
                "RandomForest": {
                    "train_r2": 0.95,
                    "test_r2": 0.91,
                },
                ...
            }
        fitted: Dict mapping model_name -> fitted model instance
                (with best hyperparameters, if GridSearch was used).

    Notes:
        - Uses R² (r2_score) as the evaluation metric.
        - If a model has an empty param grid, it will be trained directly
          without GridSearchCV.
    """
    try:
        report: Dict[str, Dict[str, float]] = {}
        fitted: Dict[str, Any] = {}

        # Loop through each model by name and instance
        for name, model in models.items():
            # Get this model's hyperparameter grid (may be empty)
            param_grid = params.get(name, {})

            # If we have a non-empty hyperparameter grid, run GridSearchCV
            if param_grid:
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=3,           # 3-fold cross-validation
                    scoring="r2",   # optimize for R²
                    n_jobs=-1,      # use all available cores
                    verbose=0,
                )
                gs.fit(X_train, y_train)

                # Use the best model found by GridSearchCV
                best_model = gs.best_estimator_
            else:
                # No hyperparameters specified: fit the model directly
                best_model = model
                best_model.fit(X_train, y_train)

            # Predictions for train and test sets
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Evaluate using R² score
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store BOTH train and test scores in the report
            report[name] = {
                "train_r2": float(train_model_score),
                "test_r2": float(test_model_score),
            }

            # Keep the fitted model so the caller can save/use it directly
            fitted[name] = best_model

        return report, fitted

    except Exception as e:
        raise CustomException(e, sys)


def load_obj(file_path: str) -> Any:
    """
    Load a previously saved Python object using dill.

    Args:
        file_path: Path to the serialized object (e.g. "artifact/model.pkl").

    Returns:
        The deserialized Python object.

    Raises:
        CustomException: If loading fails for any reason.
    """
    try:
        # "rb" = read binary mode
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
