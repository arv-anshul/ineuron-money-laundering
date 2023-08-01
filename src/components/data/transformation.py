import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.core import get_logger, io
from src.core.exception import CustomException
from src.entity.artifact import (DataIngestionArtifact,
                                 DataTransformationArtifact)
from src.entity.config import DataTransformationConfig

logger = get_logger(__name__)


@CustomException.wrap_with_custom_exception(logger)
class DataTransformation(DataTransformationConfig):
    def __init__(self, ingestion_artifact: DataIngestionArtifact):
        super().__init__()
        logger.critical(
            '%s %s %s', '>>>'*10, self.__class__.__name__, '<<<'*10,
        )
        self.ingestion = ingestion_artifact
        self.schema = self.ingestion.data_schema

    def get_transformer_object(self):
        num_pipe = Pipeline(steps=[
            ('scaler', StandardScaler()),
        ])

        obj_pipe = Pipeline(steps=[
            ('encoder', OneHotEncoder(drop='first')),
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num_pipe', num_pipe, self.schema.num_cols),
            ('obj_pipe', obj_pipe, self.schema.cat_cols),
        ])

        return preprocessor

    def initiate(self) -> DataTransformationArtifact:
        # Reading training and testing file
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        # Selecting input feature from train and test data
        X_train_df = train_df.drop(columns=[self.schema.target_name])
        X_test_df = test_df.drop(columns=[self.schema.target_name])

        # Selecting target feature from train and test data
        y_train_df = train_df[self.schema.target_name]
        y_test_df = test_df[self.schema.target_name]

        # Transformation on target columns
        target_enc = LabelEncoder()
        y_train_arr = target_enc.fit_transform(y_train_df)
        y_test_arr = target_enc.transform(y_test_df)

        preprocessor = self.get_transformer_object()
        preprocessor.fit(X_train_df)

        # Transforming input features
        X_train_arr = preprocessor.transform(X_train_df)
        X_test_arr = preprocessor.transform(X_test_df)

        train_arr = np.c_[X_train_arr, y_train_arr]
        test_arr = np.c_[X_test_arr, y_test_arr]

        # Objects dumping
        io.dump_array(train_arr, self.train_arr_path)
        io.dump_array(test_arr, self.test_arr_path)

        io.dump_model(preprocessor, self.transformer_path)
        io.dump_model(target_enc, self.target_enc_path)

        return DataTransformationArtifact(
            self.transformer_path,
            self.target_enc_path,
            self.train_arr_path,
            self.test_arr_path,
        )
