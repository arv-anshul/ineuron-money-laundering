from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.core.exception import CustomException
from src.core.logger import get_logger
from src.database import DataSchema, from_mongodb_to_dataframe
from src.entity.artifact import DataIngestionArtifact
from src.entity.config import DataIngestionConfig

logger = get_logger(__name__)


@CustomException.wrap_with_custom_exception(logger)
class DataIngestion(DataIngestionConfig):
    def __init__(self) -> None:
        super().__init__()
        logger.critical(
            '%s %s %s', '>>>'*10, self.__class__.__name__, '<<<'*10,
        )
        self.schema = DataSchema()

    def _drop_extra_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        # Check for extra columns in ingested dataset from database.
        drop_cols = [col for col in df.columns.values
                     if col not in self.schema.all_cols]
        if len(drop_cols) > 0:
            logger.info(
                'We have extra column(s) in the ingested dataset from the base dataset.'
            )
            df = df.drop(columns=drop_cols)
            logger.info('Extra column dropped: %s', drop_cols)
        return df

    def _convert_to_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        date_cols = self.schema.date_cols
        for i in date_cols:
            df[i] = pd.to_datetime(df[i])
        logger.info(
            'Conversion to datetime data type of %s column(s)', date_cols,
        )
        return df

    def _feature_extraction(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Feature extraction from DataFrame. """
        df['month'] = df[self.schema.date_cols[0]].dt.month
        df = df.drop(columns=self.schema.date_cols)

        self.schema.all_cols.append('month')
        [self.schema.all_cols.remove(i) for i in self.schema.date_cols]

        logger.info('Adding column: %s', ['month'])
        logger.info('Dropping column: %s', self.schema.date_cols)
        return df

    def initiate(
        self, ingestion_data_path: Path | None = None,
    ) -> DataIngestionArtifact:
        try:
            df = from_mongodb_to_dataframe()
        except Exception:
            logger.error('Error while importing data from database.')
            logger.info(
                'Reading "ingestion_data_path" because '
                'you have not provided "MONGODB_URL".'
            )
            if ingestion_data_path is None:
                logger.error(
                    'Please provide either "ingestion_data_path" or '
                    '"MONGODB_URL" to establish database connection.'
                )
                raise
            df = pd.read_csv(ingestion_data_path)

        df = self._drop_extra_cols(df)
        df = self._convert_to_datetime(df)
        df = self._feature_extraction(df)

        logger.info(
            'Splitting the dataset into train_df and test_df and exporting it.'
        )
        train_df, test_df = train_test_split(
            df, test_size=self.test_size, random_state=42,
        )
        logger.info('Train df shape: %s', train_df.shape)
        logger.info('Test df shape: %s', test_df.shape)

        train_df.to_csv(self.train_path, index=False)
        test_df.to_csv(self.test_path, index=False)

        return DataIngestionArtifact(
            self.base_data_path, self.schema,
            self.train_path, self.test_path,
        )
