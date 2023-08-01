from dataclasses import dataclass
from pathlib import Path

from src.core import Config
from src.core.constants import BASE_DATA_NAME

run_id = Config.get_run_id()


class PipelineConfig:
    def __init__(self):
        self.root = Path.cwd()
        self.artifact_dir = Path('artifacts', run_id)
        self.__create_all_dirs()

    def __create_all_dirs(self):
        self.artifact_dir.mkdir(parents=True, exist_ok=True)


class DataIngestionConfig(PipelineConfig):
    def __init__(self):
        super().__init__()
        self.base_data_path = self.root / 'data' / BASE_DATA_NAME
        self.dir = self.artifact_dir / 'data_ingestion'
        self.train_path = self.dir / 'train.csv'
        self.test_path = self.dir / 'test.csv'
        self.test_size = 0.2
        self.__create_all_dirs()

    def __create_all_dirs(self):
        self.dir.mkdir(parents=True, exist_ok=True)


class DataValidationConfig(DataIngestionConfig):
    def __init__(self):
        super().__init__()
        self.dir = self.artifact_dir / 'data_validation'
        self.reports_dir = self.root / 'reports'
        self.drift_report_path = self.reports_dir / 'drift_report.yaml'
        self.missing_threshold = 0.6
        self.__create_all_dirs()

    def __create_all_dirs(self):
        self.dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)


class DataTransformationConfig(DataIngestionConfig):
    def __init__(self):
        super().__init__()
        self.dir = self.artifact_dir / 'data_transformation'
        self.transformer_path = self.dir / 'transformer.pkl'
        self.target_enc_path = self.dir / 'target_encoder.pkl'
        self.train_arr_path = self.dir / 'transformed_arrays/train.npy'
        self.test_arr_path = self.dir / 'transformed_arrays/test.npy'
        self.__create_all_dirs()

    def __create_all_dirs(self):
        self.dir.mkdir(exist_ok=True)
        self.train_arr_path.parent.mkdir(parents=True, exist_ok=True)


class ModelTrainerConfig(PipelineConfig):
    def __init__(self):
        super().__init__()
        self.dir = self.artifact_dir / 'model_trainer'
        self.model_path = self.dir / 'model.pkl'
        self.expected_training_score = 0.9   # According to project
        self.expected_testing_score = 0.88   # According to project
        self.overfitting_threshold = 0.5   # According to project
        self.__create_all_dirs()

    def __create_all_dirs(self):
        self.dir.mkdir(exist_ok=True)


@dataclass
class ModelEvaluationConfig:
    model_score_diff_threshold = 0.05   # According to project


class ModelPusherConfig(PipelineConfig):
    def __init__(self):
        super().__init__()
        self.dir = self.artifact_dir / 'model_pusher'
        self.model_path = self.dir / 'model.pkl'
        self.transformer_path = self.dir / 'transformer.pkl'
        self.target_enc_path = self.dir / 'target_encoder.pkl'
        self.trained_model_dir = self.root / 'trained_models'
        self.__create_all_dirs()

    def __create_all_dirs(self):
        self.dir.mkdir(exist_ok=True)
