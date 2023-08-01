from pathlib import Path

from src.main import start_model_training

ingestion, validation = start_model_training(Path('data/base_data.csv'))

print('Done!!')
