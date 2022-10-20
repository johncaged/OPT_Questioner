import yaml
from torch_lib.util import Count


def parse_yaml(path):
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


default_config_path = './checkpoint/load.yml'


def ToCuda(item):
    return item.cuda()
    # return item.cpu()


class QuestionIdGen:
    q_id = Count()
