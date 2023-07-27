from dataclasses import dataclass
from pathlib import Path

import dill
import numpy as np

from src.core import get_logger

logger = get_logger(__name__)


@dataclass
class ProjectIO:
    """ Initialize the ProjectIO class. """

    @staticmethod
    def dump_model(model: object, fp: Path) -> None:
        with open(fp, 'wb') as f:
            dill.dump(model, f)
        logger.info('Model %s dumped.', fp)

    @staticmethod
    def load_model(fp: Path) -> object:
        with open(fp, 'rb') as f:
            logger.info('Model %s loaded.', fp)
            return dill.load(f)

    @staticmethod
    def dump_array(array: np.ndarray, fp: Path) -> None:
        array.dump(fp)
        logger.info('Array %s dumped.', fp)

    @staticmethod
    def load_array(fp: Path) -> np.ndarray:
        logger.info('Array %s loaded.', fp)
        return np.load(fp)
