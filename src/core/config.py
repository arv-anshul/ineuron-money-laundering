import json
from datetime import datetime
from pathlib import Path
from typing import Literal, TypeAlias

RunMode: TypeAlias = Literal['training', 'prediction']
RunId: TypeAlias = str
project_config_fp = Path('project.config')


class Config:
    """ Class for configuration instance attributes. """

    def create_config_file(run_mode: RunMode = 'training'):
        json.dump({
            'currentRunMode': run_mode,
            'currentRunId': datetime.now().strftime('%d%m%y_%H%M%S'),
        }, open(project_config_fp, 'w'))

    def get_run_id() -> RunId:
        """ Get the unique run ID. """
        return json.load(open(project_config_fp))['currentRunId']

    def set_run_mode(mode: RunMode) -> None:
        config_data: dict[str, str] = json.load(open(project_config_fp))
        config_data.update({'currentRunMode': mode})
        json.dump(config_data, open(project_config_fp, 'w'))

    def get_run_mode() -> RunMode:
        return json.load(open(project_config_fp))['currentRunMode']
