import json


class DataSchema:
    def __init__(self) -> None:
        schema_dict = json.load(open('src/database/schema.json'))

        self.target_name = schema_dict['targetColumn']
        self.all_cols = schema_dict['columnNames']
        self.cum_cols = schema_dict['numColumnsNames']
        self.cat_cols = schema_dict['catColumnsNames']
        self.date_cols = schema_dict['dateColumnsNames']
