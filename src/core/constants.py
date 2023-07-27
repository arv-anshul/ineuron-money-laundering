""" Contains the project constants. """

from src.database import schema_dict

BASE_DATA_NAME = 'base_data.csv'

TARGET_NAME = schema_dict['targetColumn']
ALL_COLS = schema_dict['columnNames']
NUM_COLS = schema_dict['numColumnsNames']
CAT_COLS = schema_dict['catColumnsNames']
DATE_COLS = schema_dict['dateColumnsNames']
