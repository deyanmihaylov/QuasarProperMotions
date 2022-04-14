import configparser
from pathlib import Path
import sys

from typing import Type

import Utils as U

def set_params(file_name: str) -> dict:
    """
    Set default and user-supplied parameters.
    """
    config = configparser.ConfigParser()
    config.optionxform = str # make the parser case sentitive

    set_user_params(config, file_name)
    config_parsed = eval_config_types(config)
    U.assert_config_params(config_parsed)

    return config_parsed

def set_user_params(
    config: Type[configparser.ConfigParser],
    file_name: str,
) -> None:
    """
    Parse the user-supplied parameter file.
    """

    try:
        par_file_parse_merge = config.read(file_name)
    except:
        sys.exit(f"Problem interpreting parameter file '{file_name}'.")

    if not par_file_parse_merge:
        sys.exit(f"Reading file '{file_name}' has failed.")

def eval_config_types(config: Type[configparser.ConfigParser]) -> dict:
    """
    Evaluate types of config params.
    """

    config_parsed = {}

    for section in config.sections():
        config_parsed[section] = {}

        for key in config[section]:
            try:
                config_parsed[section][key] = eval(
                    config.get(section, key)
                )
            except:
                config_parsed[section][key] = eval(
                    '"' + config.get(section, key) + '"'
                )

    return config_parsed

def record_config_params(params: dict) -> None:
    """
    Record the config parameters in a file.
    """
    
    output_params = dict()
    for group_name in params:
        output_params[group_name] = {
            k: v for k, v in params[group_name].items() if v is not None
        }

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read_dict(output_params)

    output_file_name = Path(config['General']['output_dir'], "config.par")

    with open(output_file_name, 'w') as config_file:
        config.write(config_file)

def prepare_output_dir(path_to_dir: str) -> None:
    check_and_create_dir(path_to_dir)
    check_and_create_dir(Path(path_to_dir, "log"))

def check_and_create_dir(path: str) -> None:
    """
    Check if output directory exists, and create it if it doesn't.
    """
    new_dir = Path(path).resolve()

    try:
        new_dir.mkdir(parents=True, exist_ok=True)
    except FileExistsError as e:
        sys.exit(f"Output directory could not be created. Error: '{e}'")
