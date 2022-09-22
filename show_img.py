from PIL import Image
from utils import parse_yaml, default_config_path
import os


def main(img_id, i):
    config = parse_yaml(default_config_path)
    items = config['val']
    image_name = f'{items["image_prefix"]}{img_id:012}'
    image_path = os.path.join(items['image'], f'{image_name}.jpg')
    Image.open(image_path).save(f'imgs/{i}_{image_name}.png')


if __name__ == '__main__':
    for i, img_id in enumerate([575500,
        515241,
        125547,
        493704,
        394009,
        24323,
        331807,
        482562,
        112769,
        180751,
        40842,
        388042,
        103522,
        520489,
        89487,
        288150,
        319865,
        331727,
        290076,
        228722]):
        main(img_id, i + 10)
