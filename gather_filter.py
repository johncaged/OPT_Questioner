import json
import os
from utils import QuestionIdGen
import random


def main(num, base_path='./'):
    q_id_gen = QuestionIdGen()
    
    data = []
    
    for i in range(num):
        with open(os.path.join(base_path, 'dataset_{}.json'.format(i))) as f:
            data.extend(json.load(f)['data'])
    
    for item in data:
        item['question_id'] = q_id_gen.q_id
    
    data = list(filter(lambda item: '[unused1]' not in item['question'] and '[unused1]' not in item['answer'] and item['prob'] > 0, data))
    
    with open(os.path.join(base_path, 'raw_dataset.json'), mode='w') as f:
        json.dump({'data': data}, f, indent=4)
    
    yes_cnt = len(list(filter(lambda item: item['type'] == 'yes/no' and item['answer'] == 'yes', data)))
    no_cnt = len(list(filter(lambda item: item['type'] == 'yes/no' and item['answer'] == 'no', data)))
    
    data_img, _ = create_index(data)
    
    filtered_data = []
    
    for img_id, items in data_img.items():
        yes_items = sorted(list(filter(lambda item: item['type'] == 'yes/no' and item['answer'] == 'yes', items)), key=lambda item: item['prob'], reverse=True)
        no_items = sorted(list(filter(lambda item: item['type'] == 'yes/no' and item['answer'] == 'no', items)), key=lambda item: item['prob'], reverse=True)
        number_items = sorted(list(filter(lambda item: item['type'] == 'number', items)), key=lambda item: item['prob'], reverse=True)
        other_items = sorted(list(filter(lambda item: item['type'] == 'other', items)), key=lambda item: item['prob'], reverse=True)
        
        yes_no_cnt =  1 if random.random() < 0.5 else 2
        for _ in range(yes_no_cnt):
            try:
                if random.random() < no_cnt / (yes_cnt + no_cnt):
                    filtered_data.append(yes_items.pop(0))
                else:
                    filtered_data.append(no_items.pop(0))
            except IndexError:
                if len(yes_items) > 0:
                    filtered_data.append(yes_items.pop(0))
                elif len(no_items) > 0:
                    filtered_data.append(no_items.pop(0))
                else:
                    break
        
        if random.random() < 0.5:
            filtered_data.extend(number_items[0:1])
        
        filtered_data.extend(other_items[0:2])

    yes_items = list(filter(lambda item: item['type'] == 'yes/no' and item['answer'] == 'yes', filtered_data))
    no_items = list(filter(lambda item: item['type'] == 'yes/no' and item['answer'] == 'no', filtered_data))
    number_items = list(filter(lambda item: item['type'] == 'number', filtered_data))
    other_items = list(filter(lambda item: item['type'] == 'other', filtered_data))
    
    with open(os.path.join(base_path, 'filtered_dataset.json'), 'w') as f:
        json.dump({
            'data': filtered_data,
            'count': {
                'yes/no': len(yes_items) + len(no_items),
                'number': len(number_items),
                'other': len(other_items)
            }
        }, f, indent=4)


def create_index(data):
    data_img = {}
    data_q_id = {}
    for item in data:
        data_q_id[item['question_id']] = item
        img_questions = data_img.setdefault(item['img_id'], [])
        img_questions.append(item)
    return data_img, data_q_id


if __name__ == '__main__':
    main(4)
