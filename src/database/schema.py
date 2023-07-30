import json


class DataSchema:
    def __init__(self) -> None:
        schema_dict: dict = json.load(open('src/database/schema.json'))

        self.target_name: str = schema_dict['targetColumn']
        self.all_cols: list[str] = schema_dict['columnNames']
        self.cum_cols: list[str] = schema_dict['numColumnsNames']
        self.cat_cols: list[str] = schema_dict['catColumnsNames']
        self.date_cols: list[str] = schema_dict['dateColumnsNames']
