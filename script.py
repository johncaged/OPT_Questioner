from vqa_utils.vqaTools.vqa import VQA
from utils import parse_yaml, default_config_path
from data.dataset import CC3MDataset, Tokenizer, build_cc3m_dataset, CaptionProcessor
from torch.utils.data import DataLoader
import torch
import json
from vqa_utils.vqaTools.vqa import VQA
import random
import glob


def main():
    # create data index in the format of videoOPT-Three
    
    with open('filtered_dataset.json') as f:
        data = json.load(f)['data']
    
    data_img, _ = create_index(data)

    with open('cc3m.json', 'w') as f:
        json.dump(list(data_img.keys()), f, indent=4)

    with open('txt_mapper.json', 'w') as f:
        json.dump(data_img, f, indent=4)


def main2():
    # change vqa data format to the format of videoOPT-Three
    
    vqa = VQA('./dataset/VQA/v2_mscoco_val2014_annotations.json', './dataset/VQA/v2_OpenEnded_mscoco_val2014_questions.json')
    raw_img_ids = vqa.getImgIds()
    img_ids = []
    img_data = {}
    for img_id in raw_img_ids:
        img_name = f'COCO_val2014_{img_id:012}'
        img_ids.append(img_name)
        q_ids = vqa.getQuesIds(img_id)
        for q_id in q_ids:
            answers = vqa.loadQA(q_id)[0]['answers']
            question = vqa.qqa[q_id]['question']
            _data = img_data.setdefault(img_name, [])
            for answer in answers:
                _data.append({
                    'question': question,
                    'answer': answer['answer'],
                    'question_id': q_id
                })
    
    with open('val2014std.json', 'w') as f:
        json.dump(img_ids, f, indent=4)

    with open('txt_mapper_vqa_std.json', 'w') as f:
        json.dump(img_data, f, indent=4)


def main3():
    # random split val dataset to train dataset and smaller val dataset.
    with open('dataset/COCO/train_id.json') as f:
        train_ids = json.load(f)
    train_ids = list(map(lambda item: int(item.split('_')[-1]), train_ids))
    
    with open('custom_dataset/train_id.json', 'w') as f:
        json.dump(train_ids, f, indent=4)
    
    with open('dataset/COCO/val_id.json') as f:
        val_ids = json.load(f)
    
    val_ids = list(map(lambda item: int(item.split('_')[-1]), val_ids))
    
    random.shuffle(val_ids)
    
    val_len = 2000

    with open('custom_dataset/train2_id.json', 'w') as f:
        json.dump(val_ids[val_len:], f, indent=4)
    
    with open('custom_dataset/val_id.json', 'w') as f:
        json.dump(val_ids[0:val_len], f, indent=4)


def main4():
    # check the split result
    with open('custom_dataset/train_id.json') as f:
        train_ids = json.load(f)
    
    with open('custom_dataset/train2_id.json') as f:
        train2_ids = json.load(f)
    
    with open('custom_dataset/val_id.json') as f:
        val_ids = json.load(f)
    
    train_ids = set(train_ids)
    train2_ids = set(train2_ids)
    val_ids = set(val_ids)
    
    print(len(train_ids))
    print(len(train2_ids))
    print(len(val_ids))

    print(train_ids.intersection(train2_ids))
    print(train_ids.intersection(val_ids))
    print(train2_ids.intersection(val_ids))


def main5():
    # count answers
    with open('filtered_dataset.json') as f:
        items = json.load(f)
    
    answers = set()
    for item in items['data']:
        if item['answer'] not in answers:
            answers.add(item['answer'])
    print(len(answers))


def main6():
    # count generated images
    datasets = glob.glob('dataset_[0-9]*.json')
    generated_imgs = []
    for dataset in datasets:
        with open(dataset) as f:
            data = json.load(f)
        data_img, _ = create_index(data['data'])
        generated_imgs.extend(list(map(lambda item: str(item) + '.jpg', data_img.keys())))

    with open('generated_imgs.json', 'w') as f:
        json.dump(generated_imgs, f, indent=4)


def create_index(data):
    data_img = {}
    data_q_id = {}
    for item in data:
        data_q_id[item['question_id']] = item
        img_questions = data_img.setdefault(item['img_id'], [])
        img_questions.append(item)
    return data_img, data_q_id


if __name__ == '__main__':
    main6()
