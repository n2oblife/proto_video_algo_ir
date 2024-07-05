import yaml
import numpy as np
from utils.interract import *
import Test_MavLink.camera as mlk
 

def parse_yaml_config(file_path):
    """
    Parse the YAML configuration file.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration data.
    """
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        return None

def convert_raw_temp(raw_value):
    TEMP_COEFF = 262144


if __name__ == '__main__':
    """
    Print current temperature of the camera with a given framerate and save it into a log file using:
    python Mavlink_temp.py >log\temperature.log
    """
    # PARAMETERS
    # conf_path = 'C:\Users\zKanit\Documents\Bertin_local\proto_video_algo_ir\algorithms\CompTempNUC_conf.yaml'

    # main
    config = parse_yaml_config(conf_path)
    cam = mlk.Camera(config['port_com'], config['baud_rate'])

    # TODO add a way to save the ambiant temperature with a sensor

    # TODO finish while loop
    while True:
        sys.wait()
        print(f"Temperature : {cam.temperature()}\n")
        # look at cam.com.temperature to get correct value
        if False:
            break

