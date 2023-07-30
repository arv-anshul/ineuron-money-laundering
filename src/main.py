from pathlib import Path

from src.core.config import Config # isort:skip
Config.set_run_mode('training')

from src.components import data, model


def start_model_training(ingestion_data_path: Path | None = None):
    ingestion = data.ingestion.DataIngestion().initiate(ingestion_data_path)
    validation = data.validation.DataValidation(ingestion).initiate()

    return ingestion, validation
