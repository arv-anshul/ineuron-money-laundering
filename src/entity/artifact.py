""" Entity for artifact folder to store all data and models. """

from dataclasses import dataclass
from pathlib import Path

from src.database import DataSchema


@dataclass
class DataIngestionArtifact:
    base_data_path: Path
    data_schema: DataSchema
    train_path: Path
    test_path: Path


# Maybe DataValidationArtifact is not required because it doesn't do with anything.
@dataclass
class DataValidationArtifact(DataIngestionArtifact):
    report_path: Path


@dataclass
class DataTransformationArtifact:
    transformer_path: Path
    target_enc_path: Path
    train_arr_path: Path
    test_arr_path: Path


@dataclass
class ModelTrainerArtifact:
    model_path: Path
    train_score: float
    test_score: float


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    improved_accuracy: float


@dataclass
class ModelPusherArtifact:
    dir_: Path
    trained_model_dir: Path
