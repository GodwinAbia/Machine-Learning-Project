"""
Global logging configuration for the project.

Importing this module sets up a file logger that writes to `logs/<timestamp>.log`.
Usage:
    from src.logger import logging

    logging.info("Something happened")
"""

import logging
import os
from datetime import datetime

# Name of the log file, e.g. "date_month_year_hours_minutes_seconds.log"
LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"

# Directory where all log files will be stored, e.g. "<project_root>/logs"
LOGS_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Full path to the current log file, e.g. "<project_root>/logs/date_month_year_hours_minutes_seconds.log"
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE)

# Configure the root logger to write to the log file
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] line %(lineno)d in %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


##test if logging is working
##run "python src/logging.py" in terminal

#if __name__ == "__main__":
#    logging.info("logging has started")

