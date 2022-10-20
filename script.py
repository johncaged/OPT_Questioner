from vqa_utils.vqaTools.vqa import VQA
from utils import parse_yaml, default_config_path
from data.dataset import CC3MDataset, Tokenizer, build_cc3m_dataset, CaptionProcessor
from torch.utils.data import DataLoader
import torch
import json
from vqa_utils.vqaTools.vqa import VQA


def main():
    
    def create_index(data):
        data_img = {}
        data_q_id = {}
        for item in data:
            data_q_id[item['question_id']] = item
            img_questions = data_img.setdefault(item['img_id'], [])
            img_questions.append(item)
        return data_img, data_q_id
    
    with open('filtered_dataset.json') as f:
        data = json.load(f)['data']
    
    data_img, _ = create_index(data)

    with open('cc3m.json', 'w') as f:
        json.dump(list(data_img.keys()), f, indent=4)

    with open('txt_mapper.json', 'w') as f:
        json.dump(data_img, f, indent=4)


def main2():
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


if __name__ == '__main__':
    main2()
