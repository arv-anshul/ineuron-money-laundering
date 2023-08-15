import json
from enum import Enum


class SchemaColumnType(Enum):
    ALL = '_all_cols'
    NUMERIC = '_num_cols'
    CATEGORICAL = '_cat_cols'
    DATE = '_date_cols'


class DataSchema:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataSchema, cls).__new__(cls)
            cls._instance._load_schema()
        return cls._instance

    def _load_schema(self):
        schema_dict = json.load(open('src/database/schema.json'))

        self._target_name: str = schema_dict['targetColumn']
        self._all_cols: list[str] = schema_dict['columnNames']
        self._num_cols: list[str] = schema_dict['numColumnsNames']
        self._cat_cols: list[str] = schema_dict['catColumnsNames']
        self._date_cols: list[str] = schema_dict['dateColumnsNames']

    @property
    def target_name(self) -> str:
        return self._target_name

    @property
    def all_cols(self) -> list[str]:
        return self._all_cols

    @property
    def num_cols(self) -> list[str]:
        return self._num_cols

    @property
    def cat_cols(self) -> list[str]:
        return self._cat_cols

    @property
    def date_cols(self) -> list[str]:
        return self._date_cols

    def update_column(self, col_type: SchemaColumnType, col: str | list[str]) -> None:
        col_list: list[str] = getattr(self, col_type.value)
        if isinstance(col, str):
            col_list.append(col)
        elif isinstance(col, list):
            col_list.extend(col)

    def remove_value_from_column(self, col_type: SchemaColumnType, col: str | list[str]) -> None:
        col_list: list[str] = getattr(self, col_type.value)
        if isinstance(col, str):
            col_list.remove(col)
        if isinstance(col, list):
            [col_list.remove(i) for i in col]
