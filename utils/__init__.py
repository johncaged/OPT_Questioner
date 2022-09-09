import yaml


def parse_yaml(path):
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


default_config_path = './checkpoint/load.yml'
