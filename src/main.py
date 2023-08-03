from pathlib import Path
from typing import Any

import pandas as pd

from src.core.config import Config  # isort:skip
Config.set_run_mode('training')

from src.components import data, model
from src.core import io
from src.entity.saved_model import SavedModelConfig


def start_model_training(ingestion_data_path: Path | None = None):
    ingestion = data.ingestion.DataIngestion().initiate(ingestion_data_path)
    data.validation.DataValidation(ingestion).initiate()
    transformation = data.transformation.DataTransformation(
        ingestion).initiate()
    trainer = model.trainer.ModelTrainer().initiate()
    model.evaluation.ModelEvaluation(
        ingestion, transformation, trainer).initiate()


def get_latest_models():
    saved_model_config = SavedModelConfig()
    latest_models = saved_model_config.get_saved_models_path()

    model = io.load_model(latest_models[0])
    transformer = io.load_model(latest_models[1])
    target_enc = io.load_model(latest_models[2])
    return model, transformer, target_enc


def predict(df: pd.DataFrame) -> tuple[pd.DataFrame, Any]:
    model, transformer, target_enc = get_latest_models()
    input_arr = transformer.transform(df[transformer.feature_names_in_])
    prediction = model.predict(input_arr)
    prediction = target_enc.inverse_transform(prediction.astype(int))
    df['prediction'] = prediction
    return df, prediction
