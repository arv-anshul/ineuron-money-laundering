from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score

from src.core import CustomException, get_logger, io
from src.entity.artifact import (DataIngestionArtifact,
                                 DataTransformationArtifact,
                                 ModelEvaluationArtifact, ModelTrainerArtifact)
from src.entity.config import ModelEvaluationConfig
from src.entity.saved_model import SavedModelConfig

logger = get_logger(__name__)


@CustomException.wrap_with_custom_exception(logger)
class ModelEvaluation(ModelEvaluationConfig):
    def __init__(
        self,
        ingestion_artifact: DataIngestionArtifact,
        transformation_artifact: DataTransformationArtifact,
        trainer_artifact: ModelTrainerArtifact,
    ) -> None:
        super().__init__()
        logger.critical(
            '%s %s %s', '>>>'*10, self.__class__.__name__, '<<<'*10,
        )

        self.ingestion_artifact = ingestion_artifact
        self.schema = ingestion_artifact.data_schema
        self.transformer_artifact = transformation_artifact
        self.trainer_artifact = trainer_artifact
        self.saved_models = SavedModelConfig()

    def __load_saved_objects(
        self, model_fp, transformer_fp,
    ) -> tuple[Any, Any]:
        model = io.load_model(model_fp)
        transformer = io.load_model(transformer_fp)
        return model, transformer

    def __copy_to_path(self, from_: Path, to: Path) -> None:
        with open(from_, 'rb') as f:
            with open(to, 'wb') as x:
                x.write(f.read())

    def __dump_models_into_saved_models_dir(self) -> None:
        from_paths = [
            self.trainer_artifact.model_path,
            self.transformer_artifact.transformer_path,
            self.transformer_artifact.target_enc_path,
        ]
        to_paths = [
            self.saved_models.path_to_save_model,
            self.saved_models.path_to_save_transformer,
            self.saved_models.path_to_save_target_enc,
        ]

        for from_, to in zip(from_paths, to_paths):
            from_.parent.mkdir(parents=True, exist_ok=True)
            to.parent.mkdir(parents=True, exist_ok=True)

            logger.info('File: "%s" copying to "%s"', from_, to)
            self.__copy_to_path(from_, to)

    def initiate(self) -> ModelEvaluationArtifact:
        if self.saved_models.latest_saved_dir is None:
            logger.info(
                'There are no Pre-Trained model. '
                'So this is the first trained model.'
            )
            self.__dump_models_into_saved_models_dir()
            return ModelEvaluationArtifact(True, 0.0)

        logger.info('Pre-Trained model found. Initialize ModelEvaluation.')

        logger.info('Importing saved trained objects.')
        model, transformer = self.__load_saved_objects(
            self.saved_models.path_to_save_model,
            self.saved_models.path_to_save_transformer,
        )

        logger.info('Newly trained model objects')
        new_model, new_transformer = self.__load_saved_objects(
            self.trainer_artifact.model_path,
            self.transformer_artifact.transformer_path,
        )

        # --- --- Old Model Evaluation --- --- #
        logger.info('%s Old Model Evaluation %s', '==='*10, '==='*10)
        test_df = pd.read_csv(self.ingestion_artifact.test_path)
        y_true = test_df[self.schema.target_name]

        input_arr = transformer.transform(
            test_df[transformer.get_feature_names_out()]
        )
        y_pred = model.predict(input_arr)

        old_score = accuracy_score(y_true, y_pred)
        logger.info('Score of old model: %s', old_score)

        # --- --- New Model Evaluation --- --- #
        logger.info('%s New Model Evaluation %s', '==='*10, '==='*10)
        input_arr = new_transformer.transform(
            test_df[transformer.get_feature_names_out()]
        )
        y_pred = new_model.predict(input_arr)

        current_score = accuracy_score(y_true, y_pred)
        logger.info('Score of current model: %s', current_score)

        if current_score <= old_score:
            msg = 'New trained model is not better than old model.'
            logger.error(msg)
            raise ValueError(msg)

        logger.info(
            'New model is better than old model. So save the new model.'
        )
        self.__dump_models_into_saved_models_dir()
        return ModelEvaluationArtifact(
            True,
            current_score-old_score,    # type: ignore
        )
