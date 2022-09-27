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
    for i, img_id in enumerate([75563]):
        main(img_id, i)
