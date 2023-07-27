import logging
from dataclasses import dataclass
from pathlib import Path

from src.core import Config


def get_logger(logger_name: str) -> logging.Logger:
    """
    :logger_name (str): `__name__`

    :returns: logging.Logger
    """
    return Logger(logger_name).get_logger


@dataclass
class Logger:
    """
    Logger for the project.

    Args:
        logger_name: __name__
    """

    logger_name: str

    def __post_init__(self):
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(logging.DEBUG)

        self.run_mode = Config.get_run_mode()
        self.run_id = Config.get_run_id()

        fp = Path(f'logs/{self.run_mode}_logs/{self.run_id}.log')
        fp.parent.mkdir(parents=True, exist_ok=True)

        formatter = logging.Formatter(
            "[ %(asctime)s ] %(filename)s:[%(lineno)d] - %(name)s - %(levelname)s - %(message)s"
        )

        file_handler = logging.FileHandler(fp)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    @property
    def get_logger(self) -> logging.Logger:
        """ Get the Logger object. """
        return self.logger
