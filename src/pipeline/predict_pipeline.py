"""
Prediction pipeline.

- PredictPipeline: loads the trained model and preprocessor, and runs predictions.
- CustomData: maps raw user inputs (e.g. from a web form) into a DataFrame
  that matches the training feature schema.
"""

import os
import sys
from typing import Any

import pandas as pd

from src.exception import CustomException
from src.utils import load_obj


class PredictPipeline:
    """
    Pipeline responsible for loading the trained model and preprocessor
    and using them to generate predictions on incoming feature data.
    """

    def __init__(
        self,
        model_path: str | None = None,
        preprocessor_path: str | None = None,
    ) -> None:
        """
        Initialize the prediction pipeline.

        Args:
            model_path: Optional custom path to the trained model pickle file.
                        Defaults to "artifact/model.pkl".
            preprocessor_path: Optional custom path to the preprocessor pickle file.
                               Defaults to "artifact/preprocessor.pkl".
        """
        self.model_path = model_path or os.path.join("artifact", "model.pkl")
        self.preprocessor_path = preprocessor_path or os.path.join(
            "artifact", "preprocessor.pkl"
        )

    def predict(self, features: pd.DataFrame) -> Any:
        """
        Run predictions on the given feature data.

        Args:
            features: A pandas DataFrame containing the same columns used
                      during training (e.g. output of CustomData.get_data_as_frame()).

        Returns:
            The model predictions (usually a numpy array or list-like).

        Raises:
            CustomException: If loading artifacts or prediction fails.
        """
        try:
            # Load model and preprocessor
            model = load_obj(file_path=self.model_path)
            preprocessor = load_obj(file_path=self.preprocessor_path)

            # Transform input features using the fitted preprocessor
            data_scaled = preprocessor.transform(features)

            # Predict using the trained model
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Container for input features required by the model.

    Responsible for:
    - Storing user-provided values (e.g. from an HTML form).
    - Converting them into a pandas DataFrame that matches the training
      feature schema (same column names as the original dataset).
    """

    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ) -> None:
        """
        Initialize the CustomData object with raw feature values.

        Args:
            gender: Student's gender.
            race_ethnicity: Group label (e.g. "group A", "group B", ...).
            parental_level_of_education: Parent's highest education level.
            lunch: Lunch type (e.g. "standard", "free/reduced").
            test_preparation_course: Whether test prep course was completed.
            reading_score: Reading score (integer).
            writing_score: Writing score (integer).
        """
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_frame(self) -> pd.DataFrame:
        """
        Convert stored feature values into a single-row pandas DataFrame.

        Returns:
            A DataFrame with columns matching the training data:
                - "gender"
                - "race/ethnicity"
                - "parental level of education"
                - "lunch"
                - "test preparation course"
                - "reading score"
                - "writing score"

        Raises:
            CustomException: If DataFrame construction fails.
        """
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [
                    self.parental_level_of_education
                ],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)