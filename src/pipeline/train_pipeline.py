"""
Training pipeline orchestration.

This module wires together:
- DataIngestion
- DataTransformation
- ModelTrainer

and runs the full end-to-end training pipeline.

Run from the project root with:
    python -m src.pipeline.train_pipeline
"""

import sys

from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def run_training_pipeline() -> tuple[str, float]:
    """
    Run the full training pipeline: ingestion -> transformation -> training.

    Returns:
        A tuple of:
            (best_model_name, final_test_r2_score)

    Raises:
        CustomException: If any step of the pipeline fails.
    """
    try:
        logging.info("Starting training pipeline.")

        # 1) Data ingestion
        ingestion = DataIngestion()
        train_data_path, test_data_path = ingestion.initiate_data_ingestion()
        logging.info(
            f"Data ingestion completed. "
            f"Train: {train_data_path}, Test: {test_data_path}"
        )

        # 2) Data transformation
        data_transformation = DataTransformation()
        train_array, test_array, _ = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        logging.info("Data transformation completed.")

        # 3) Model training
        model_trainer = ModelTrainer()
        best_model_name, model_score = model_trainer.initiate_model_trainer(
            train_array, test_array
        )
        logging.info(
            f"Model training completed. "
            f"Best model: {best_model_name}, Final test R²: {model_score:.4f}"
        )

        return best_model_name, model_score

    except Exception as e:
        # Wrap and log any exception from the pipeline
        raise CustomException(e, sys)



if __name__ == "__main__":
    """
    Entry point for running the training pipeline as a script.
    """
    try:
        best_model_name, score = run_training_pipeline()
        print("Training pipeline finished.")
        print(f"Best model: {best_model_name}")
        print(f"Test R² score: {score:.4f}")
    except CustomException as e:
        # Already logged inside CustomException; just print to console as well.
        print(f"Training pipeline failed: {e}")


# Run this from the terminal (project root):
#   python -m src.pipeline.train_pipeline