"""
Data ingestion component.

- Reads the raw dataset from disk.
- Splits it into train/test sets.
- Saves raw, train, and test CSVs into the artifact directory.
"""

import os
import sys
from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    """
    Configuration for data ingestion.

    Attributes:
        train_data_path: Path where the training split CSV will be saved.
        test_data_path: Path where the test split CSV will be saved.
        raw_data_path: Path where the raw input CSV will be saved.
    """
    train_data_path: str = os.path.join("artifact", "train.csv")
    test_data_path: str = os.path.join("artifact", "test.csv")
    raw_data_path: str = os.path.join("artifact", "raw.csv")


class DataIngestion:
    """
    Data ingestion pipeline step.

    Responsible for:
    - Reading the source dataset.
    - Creating train/test splits.
    - Saving them to the configured artifact paths.

    keep in mind:
    If CSV moves or change its location, 
    you’ll just need to update source_data_path or pass it explicitly when you create DataIngestion
    """

    #change this line of code to read in data from different places
    def __init__(self, source_data_path: str = "notebook/StudentsPerformance.csv") -> None:
        """
        Initialize the DataIngestion component.

        Args:
            source_data_path: Path to the raw dataset to be ingested.
        """
        self.ingestion_config = DataIngestionConfig()
        self.source_data_path = source_data_path

    def initiate_data_ingestion(self) -> Tuple[str, str]:
        """
        Execute the data ingestion process.

        Returns:
            (train_data_path, test_data_path)

        Raises:
            CustomException: If any step fails.
        """
        logging.info("Entered the data ingestion method.")

        try:
            logging.info(f"Reading dataset from: {self.source_data_path}")
            #change this line of code to read in data from different places
            df = pd.read_csv(self.source_data_path)
            logging.info("Read raw data into DataFrame successfully.")

            artifact_dir = os.path.dirname(self.ingestion_config.train_data_path)
            os.makedirs(artifact_dir, exist_ok=True)
            logging.info(f"Ensured artifact directory exists at: {artifact_dir}")

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved to: {self.ingestion_config.raw_data_path}")

            logging.info("Performing train-test split (test_size=0.2, random_state=42).")
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42
            )

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info(
                "Data ingestion completed successfully. "
                f"Train data: {self.ingestion_config.train_data_path}, "
                f"Test data: {self.ingestion_config.test_data_path}"
            )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
        
"""
run this in terminal to "python -m src.pipeline.train_pipeline" to run ingestion, 
create artifact folder + CSVs, also trigger trnasformation + trianing and do loggings
"""