import json
from datetime import datetime
from pathlib import Path
from typing import Literal, TypeAlias

RunMode: TypeAlias = Literal['training', 'prediction']
RunId: TypeAlias = str
project_config_fp = Path('project.config')


class Config:
    """ Class for configuration instance attributes. """

    @staticmethod
    def _create_config_file(run_mode: RunMode = 'training'):
        if not project_config_fp.exists():
            json.dump({
                'currentRunMode': run_mode,
                'currentRunId': datetime.now().strftime('%d%m%y_%H%M%S'),
            }, open(project_config_fp, 'w'))
        else:
            raise FileExistsError(
                'Use Config.set_run_mode() to set RunMode.'
            )

    @staticmethod
    def get_run_id() -> RunId:
        """ Get the unique run ID. """
        return json.load(open(project_config_fp))['currentRunId']

    @staticmethod
    def _update_run_id() -> None:
        """ Update the RunID. """
        data: dict = json.load(open(project_config_fp))
        data.update({'currentRunId': datetime.now().strftime('%d%m%y_%H%M%S')})
        json.dump(data, open(project_config_fp, 'w'))

    @staticmethod
    def set_run_mode(mode: RunMode) -> None:
        """ Set RunMode for the project. """
        if not project_config_fp.exists():
            Config._create_config_file(mode)

        Config._update_run_id()
        config_data: dict = json.load(open(project_config_fp))
        config_data.update({'currentRunMode': mode})
        json.dump(config_data, open(project_config_fp, 'w'))

    @staticmethod
    def get_run_mode() -> RunMode:
        return json.load(open(project_config_fp))['currentRunMode']
