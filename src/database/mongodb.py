import json
from enum import Enum
from os import getenv

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection

from src.core import get_logger

load_dotenv()
logger = get_logger(__name__)


class MongoDB(Enum):
    database_name = 'money_laundering'
    collection_name = 'base_data'


def get_mongodb_url() -> str:
    mongodb_url = getenv('MONGODB_URL')
    try:
        if mongodb_url is None:
            import streamlit as st
            mongodb_url = st.secrets['MONGODB_URL']
            logger.info('Getting MONGODB_URL from streamlit secrets.')
        else:
            logger.info('Getting MONGODB_URL from .env file.')
    except Exception:
        logger.error(
            'No env_var is available as MONGODB_URL in ".env" or '
            '".streamlit/secrets.toml" file.'
        )
        raise
    else:
        return mongodb_url


def get_collection_connection(
    mongodb_url: str, database_name: MongoDB, collection_name: MongoDB,
) -> Collection:
    client = MongoClient(mongodb_url)
    logger.info('Connection with MONGODB is established.')
    return client[database_name.value][collection_name.value]


def dump_data_to_mongodb(df: pd.DataFrame) -> None:
    """ Dump your `base_data` to MongoDB. """
    mongodb_url = get_mongodb_url()
    coll = get_collection_connection(
        mongodb_url, MongoDB.database_name, MongoDB.collection_name,
    )

    data_as_list: list[dict] = list(json.loads(df.T.to_json()).values)
    coll.insert_many(data_as_list)
    logger.info('Dumped %s data into MONGODB.', len(data_as_list))


def from_mongodb_to_dataframe() -> pd.DataFrame:
    """
    Get your `base_data` from MongoDB as DataFrame.

    DataFrame contains the `_id` column so you need to `drop` it.
    """
    mongodb_url = get_mongodb_url()
    coll = get_collection_connection(
        mongodb_url, MongoDB.database_name, MongoDB.collection_name,
    )

    base_data = coll.find()
    df = pd.DataFrame(base_data)
    logger.info('Loaded %s shaped DataFrame from MONGODB.', df.shape)
    return df


if __name__ == '__main__':
    df = pd.read_csv('data/raw_data.csv')
    print(df.shape)
    # dump_data_to_mongodb(df)
