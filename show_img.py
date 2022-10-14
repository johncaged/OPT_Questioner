from PIL import Image
from utils import parse_yaml, default_config_path
import os


def main(img_id, i):
    config = parse_yaml(default_config_path)
    prefix = config['cc3m']['img_path']
    image_name = f'{img_id}.jpg'
    image_path = os.path.join(prefix, f'{image_name}')
    Image.open(image_path).save(f'imgs/{i}_{image_name}')


if __name__ == '__main__':
    for i, img_id in enumerate([
        376541917,
        1479978390,
        1460098297,
        634739811
    ]):
        main(img_id, i)
