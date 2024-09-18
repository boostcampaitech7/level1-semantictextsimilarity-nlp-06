import yaml
from box import Box

def load_config(config_file):
    # Load Config.yaml
    with open(config_file) as file:
        config = yaml.safe_load(file)
        config = Box(config)

    return config