""" Saved Model Entity to track recently saved trained model. """

from pathlib import Path

from src.core import get_logger

logger = get_logger(__name__)
saved_model_dir = Path('saved_models')


class SavedModelConfig:
    def __init__(self) -> None:
        self.dir = saved_model_dir
        self.dir.mkdir(exist_ok=True)

        self.latest_saved_dir = self.__get_latest_saved_dir_path()
        self.new_dir_to_save_models = self.__get_new_dir_path_to_save()

    def __get_latest_saved_dir_path(self) -> Path | None:
        dir_names = [int(i.name) for i in self.dir.iterdir()]
        if len(dir_names) == 0:
            return None
        return self.dir / str(max(dir_names))

    def __get_new_dir_path_to_save(self) -> Path:
        if self.latest_saved_dir == None:
            return self.dir / str(0)
        latest_dir_number = int(self.latest_saved_dir.name) + 1
        return self.dir / str(latest_dir_number)

    def get_saved_models_path(self) -> tuple[Path, Path, Path]:
        latest = self.latest_saved_dir
        if latest is None:
            raise FileNotFoundError('There is no saved models.')
        return (
            latest / 'model.pkl',
            latest / 'transformer.pkl',
            latest / 'target_encoder.pkl',
        )

    @property
    def path_to_save_model(self):
        return self.new_dir_to_save_models / 'model.pkl'

    @property
    def path_to_save_transformer(self):
        return self.new_dir_to_save_models / 'transformer.pkl'

    @property
    def path_to_save_target_enc(self):
        return self.new_dir_to_save_models / 'target_encoder.pkl'
