"""
Custom exception handling.

This module defines:
- error_message_detail: build a error message with file/line info.
- CustomException: wrap any exception, get extra context and logs it.
"""


import sys
from typing import Any

from src.logger import logging


def error_message_detail(error: Exception, error_detail: Any) -> str:
    """
    Build a detailed error message with script filename and line number.

    Args:
        error: The original exception object.
        error_detail: Typically the `sys` module, used to access exc_info().

    Returns:
        A formatted string describing where the error occurred and why.
    """

    # exc_info() returns (exc_type, exc_value, traceback)
    _, _, exc_tb = error_detail.exc_info()

    # Fallback in case there is no traceback (rare but safe to handle)
    if exc_tb is None:
        return f"Unhandled error: {error!r}"

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    error_message = (
        f"Error occurred in python script [{file_name}], "
        f"line [{line_number}], "
        f"message: {error!r}"
    )
    return error_message


class CustomException(Exception):
    """
    Custom exception.

    Typical usage:
        try:
            ...
        except Exception as e:
            raise CustomException(e, sys)

    It wraps the original exception, attaches file/line info, and logs the error.
    """

    def __init__(self, error: Exception, error_detail: Any) -> None:
        """
        Initialize the CustomException with detailed context.

        Args:
            error: The original exception instance.
            error_detail: Typically the `sys` module, used to access exc_info().
        """
        message = error_message_detail(error, error_detail)
        super().__init__(message)
        self.error_message = message

        # Log the error immediately using the global logger
        logging.error(self.error_message)

    def __str__(self) -> str:
        """Return the detailed error message when the exception is printed."""
        return self.error_message

    def __repr__(self) -> str:
        """Official string representation, useful in logs and debugging."""
        return f"{self.__class__.__name__}({self.error_message!r})"


#testing if it works exception handling works

"""
if __name__ == "__main__":
    try:
        a=1/0
    except Exception as e:
        logging.info("divide error")
        raise CustomException(e,sys)   
"""
