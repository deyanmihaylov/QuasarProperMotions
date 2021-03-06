import configparser
import os
import sys

from typing import Type

def set_params(
        file_name: str
    ) -> dict:
    """
    Set default and user-supplied parameters.
    """

    config = configparser.ConfigParser()
    config.optionxform = str # make the parser case sentitive

    set_default_params(config)

    set_user_params(config, file_name)

    config_parsed = eval_config_types(config)

    return config_parsed

def set_default_params(
        config: Type[configparser.ConfigParser]
    ) -> None:
    """
    Parse the default parameter file.
    """

    if '__file__' not in globals():
        sys.exit("Global variable __file__ is not set. Default values for parameters cannot be set.")
    else:
        code_env = os.path.realpath(__file__)
        code_dir = os.path.dirname(code_env)

        default_file_name = os.path.join(code_dir, "par/default.par")

        try:
            config.read(default_file_name)
        except:
            sys.exit("Problem reading default parameter file.")

def set_user_params(
        config: Type[configparser.ConfigParser],
        file_name: str
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

def eval_config_types(
        config: Type[configparser.ConfigParser]
    ) -> dict:
    """
    Evaluate types of config params.
    """

    config_parsed = dict()

    for section in config.sections():
        config_parsed[section] = dict()

        for key in config[section]:
            try:
                config_parsed[section][key] = eval(config.get(section, key))
            except:
                config_parsed[section][key] = eval('"' + config.get(section, key) + '"')

    return config_parsed

def check_output_dir(
        dir_name: str
    ) -> bool:
    """
    Check if output directory exists, and create it if it doesn't.
    """

    if not os.path.isdir(dir_name):
        try:
            os.makedirs(dir_name, exist_ok=True)
        except OSError:
            sys.exit("Output directory could not be created. Error: '{OSError}'")

    return True

def record_config_params(
        params: dict,
        user_specified_output_file = None
    ) -> None:
    """
    Record the config parameters in a file.
    """
    
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read_dict(params)

    if user_specified_output_file:
        output_file_name = user_specified_output_file
    else:
        output_file_name = os.path.join(config['General']['output_dir'], "config.par")

    with open(output_file_name, 'w') as config_file:
        config.write(config_file)
