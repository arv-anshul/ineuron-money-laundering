from src.components import data, model
from pathlib import Path

from src.core.config import Config  # isort:skip
Config.set_run_mode('training')


def start_model_training(ingestion_data_path: Path | None = None):
    ingestion = data.ingestion.DataIngestion().initiate(ingestion_data_path)
    validation = data.validation.DataValidation(ingestion).initiate()
    transformation = data.transformation.DataTransformation(
        ingestion).initiate()
    trainer = model.trainer.ModelTrainer().initiate()
    evaluation = model.evaluation.ModelEvaluation(
        ingestion, transformation, trainer).initiate()

    return ingestion, validation
