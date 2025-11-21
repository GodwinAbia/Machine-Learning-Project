"""
Data transformation component.

- Builds preprocessing pipelines for numeric and categorical features.
- Applies transformations to train and test data.
- Saves the fitted preprocessor object to disk for reuse (e.g. in prediction).
"""

import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation.

    Attributes:
        preprocessor_obj_file_path: Path where the fitted preprocessing
            object (e.g. ColumnTransformer) will be saved as a pickle file.
    """
    preprocessor_obj_file_path: str = os.path.join("artifact", "preprocessor.pkl")


class DataTransformation:
    """
    Data transformation pipeline step.

    Responsible for:
    - Creating preprocessing pipelines for numeric and categorical features.
    - Fitting the preprocessor on training data and transforming both
      train and test sets.
    - Saving the fitted preprocessor for later use (e.g. in prediction).
    """

    def __init__(self) -> None:
        """
        Initialize the DataTransformation component with its configuration.
        """
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self) -> ColumnTransformer:
        """
        Build and return the preprocessing ColumnTransformer.

        The transformer:
        - Numeric features: median imputation + StandardScaler.
        - Categorical features: most_frequent imputation + OneHotEncoder
          + StandardScaler(with_mean=False).

        Returns:
            A scikit-learn ColumnTransformer that can be fit/transformed.

        Raises:
            CustomException: If construction of the transformer fails.
        """
        try:
            numeric_features = ["writing score", "reading score"]
            categorical_features = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]

            # Pipeline for numeric features: impute missing values, then scale
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            # Pipeline for categorical features: impute, one-hot encode, then scale
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info(f"Numeric columns for scaling: {numeric_features}")
            logging.info(f"Categorical columns for encoding: {categorical_features}")

            # Combine numeric and categorical pipelines into a single transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numeric_features),
                    ("cat_pipeline", cat_pipeline, categorical_features),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self,
        train_path: str,
        test_path: str,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Execute the data transformation process.

        Steps:
        - Read train and test CSVs from the given paths.
        - Split features and target.
        - Fit the preprocessor on the training features.
        - Transform both train and test features.
        - Concatenate transformed features with the target into arrays.
        - Save the fitted preprocessor object to disk.

        Args:
            train_path: Path to the training CSV file.
            test_path: Path to the test CSV file.

        Returns:
            A tuple of:
                (train_array, test_array, preprocessor_obj_file_path)
            where train_array and test_array are numpy arrays with the
            transformed features and target as the last column.

        Raises:
            CustomException: If any step in the transformation process fails.
        """
        try:
            # Read datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test datasets have been loaded.")
            logging.info("Obtaining preprocessing object.")

            preprocessor_obj = self.get_data_transformer_obj()

            target_column_name = "math score"
            numeric_features = ["writing score", "reading score"]  # kept for clarity

            # Separate input features and target for training data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate input features and target for test data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and test dataframes.")

            # Fit on training features, transform both train and test
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            # Concatenate the target feature as the last column
            train_array = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_array = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Preprocessing complete. Saving preprocessor object.")

            # Ensure directory for preprocessor exists
            preprocessor_dir = os.path.dirname(
                self.data_transformation_config.preprocessor_obj_file_path
            )
            if preprocessor_dir:
                os.makedirs(preprocessor_dir, exist_ok=True)

            # Save the fitted preprocessor object for later use
            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj,
            )

            logging.info(
                f"Preprocessor object saved at: "
                f"{self.data_transformation_config.preprocessor_obj_file_path}"
            )

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
