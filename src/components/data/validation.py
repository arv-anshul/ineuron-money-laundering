import json
from warnings import warn

import pandas as pd
from pandas import DataFrame
from scipy.stats import ks_2samp

from src.core import get_logger
from src.database.schema import DataSchema
from src.entity.artifact import DataIngestionArtifact, DataValidationArtifact
from src.entity.config import DataValidationConfig

logger = get_logger(__name__)


class DataValidation(DataValidationConfig):
    def __init__(self, ingestion_artifact: DataIngestionArtifact):
        """
        Initiate validation process between base, train and test dataset.
        """
        super().__init__()
        logger.critical('%s %s %s', '>>>' * 10, self.__class__.__name__, '<<<' * 10)
        self.ingestion_artifact = ingestion_artifact
        self.schema = DataSchema()
        self.validation_report = {}

    def _drop_missing_values_cols(
        self,
        df: DataFrame,
        dataset_type: str,
    ) -> DataFrame | None:
        """
        This function will drop column which contains missing value more than specified threshold.

        Returns
        ------
        DataFrame if at least a single column is available after dropping missing columns else None.
        """
        logger.info('Checking missing values in %s.', dataset_type)

        report_name = 'missing_values_within_' + dataset_type
        threshold = self.missing_threshold
        null_report = df.isna().sum().div(df.shape[0])

        drop_cols = list(null_report[null_report > threshold].index)
        self.validation_report[report_name] = drop_cols

        if df.shape[1] == 0:
            logger.info('Dropping %s columns from %s.', drop_cols, dataset_type)
            return None

        return df

    def _is_required_cols_exists(
        self,
        curr_df: DataFrame,
        dataset_type: str,
    ) -> bool:
        report_name = 'missing_cols_within_' + dataset_type

        missing_cols = list(set(self.schema.all_cols) - set(curr_df.columns))
        logger.info('Column: %s is not available in %s.', missing_cols, dataset_type)

        if len(missing_cols) > 0:
            msg = f'Columns are missing from {dataset_type}: {missing_cols}'
            logger.warning(msg)
            warn(msg)
            self.validation_report[report_name] = missing_cols
            return False
        return True

    def _data_drift(
        self,
        base_df: DataFrame,
        curr_df: DataFrame,
        dataset_type: str,
    ) -> None:
        report_name = 'data_drift_within_' + dataset_type
        drift_report = {}
        for base_col in self.schema.all_cols:
            try:
                base_data = base_df[base_col]
            except KeyError:
                msg = (
                    f'"{base_col}" is not a column in {dataset_type}. '
                    f'Maybe "{base_col}" is a new column in schema.'
                )
                logger.error(msg)
                warn(msg, FutureWarning)
                continue
            curr_data = curr_df[base_col]

            logger.info('Hypothesis "%s": %s, %s', base_col, base_data.dtype, curr_data.dtype)
            distribution = ks_2samp(base_data, curr_data)
            pvalue = float(distribution.pvalue)  # type: ignore

            if pvalue > 0.05:
                drift_report[base_col] = {'pvalue': round(pvalue, 3), 'same_distribution': True}
            else:
                drift_report[base_col] = {'pvalue': round(pvalue, 3), 'same_distribution': False}
        self.validation_report[report_name] = drift_report

    def initiate(self) -> DataValidationArtifact:
        # --- --- Base Dataset --- --- #
        logger.info('Reading base dataset.')
        base_df = pd.read_csv(self.base_data_path)
        base_df = self._drop_missing_values_cols(base_df, 'base_df')

        # --- --- Train dataset --- --- #
        logger.info('Reading train dataset.')
        train_df = pd.read_csv(self.train_path)
        train_df = self._drop_missing_values_cols(train_df, 'train_df')

        # --- --- Test Dataset --- --- #
        logger.info('Reading test dataset.')
        test_df = pd.read_csv(self.test_path)
        test_df = self._drop_missing_values_cols(test_df, 'test_df')

        # --- --- Check datasets exists --- --- #
        if base_df is None:
            raise ValueError('Base Dataset cannot be None.')
        if train_df is None:
            raise ValueError('Train Dataset cannot be None.')
        if test_df is None:
            raise ValueError('Test Dataset cannot be None.')

        # --- --- Checking Data Drift --- --- #
        drift_log_msg = 'All columns are available in %s. Hence calculating data drift.'

        # --- --- Train Dataset --- --- #
        train_df_cols_status = self._is_required_cols_exists(train_df, 'train_df')
        if train_df_cols_status:
            logger.info(drift_log_msg % 'train_df')
            self._data_drift(base_df, train_df, 'train_df')

        # --- --- Test Dataset --- --- #
        test_df_cols_status = self._is_required_cols_exists(test_df, 'test_df')
        if test_df_cols_status:
            logger.info(drift_log_msg % 'test_df')
            self._data_drift(base_df, test_df, 'test_df')

        json.dump(self.validation_report, open(self.drift_report_path, 'w'), indent=2)

        return DataValidationArtifact(
            self.base_data_path,
            self.train_path,
            self.test_path,
            self.drift_report_path,
        )
