import json
import os
from utils import QuestionIdGen
import random
import glob
from tqdm import tqdm


def main(base_path='./'):
    data = gather_data(base_path)
    
    yes_cnt = len(list(filter(lambda item: item['type'] == 'yes/no' and item['answer'] == 'yes', data)))
    no_cnt = len(list(filter(lambda item: item['type'] == 'yes/no' and item['answer'] == 'no', data)))
    
    data_img, _ = create_index(data)
    
    print('generated imgs: {}'.format(len(data_img)))
    
    filtered_data = []
    
    for img_id, items in tqdm(data_img.items()):
        yes_items, no_items, number_items, other_items = separate_by_category(items)
        yes_items = sort_answer(yes_items)
        no_items = sort_answer(no_items)
        number_items = sort_answer(number_items)
        other_items = sort_answer(other_items)
        
        yes_no_cnt = 1 if random.random() < 0.5 else 2
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
        count_and_save(filtered_data, base_path)


def main2(batch_size=256, base_path='./'):
    data = gather_data(base_path)
    
    data_img, _ = create_index(data)
    print('generated imgs: {}'.format(len(data_img)))
    
    yes_pick_count = int(4 * batch_size * 0.75 / 4)
    no_pick_count = int(4 * batch_size * 0.75 / 4)
    number_pick_count = int(4 * batch_size * 0.5 / 4)
    other_pick_count = int(4 * batch_size * 2 / 4)
    
    filtered_data = []
    batch_data = []
    print('filtering data...')
    for i, items in enumerate(tqdm(data_img.values())):
        batch_data.extend(items)
        if (i + 1) % batch_size == 0 or (i + 1) == len(data_img):
            yes_items, no_items, number_items, other_items = separate_by_category(batch_data)
            
            yes_items = sort_answer(yes_items)[0:yes_pick_count]
            no_items = sort_answer(no_items)[0:no_pick_count]
            number_items = sort_answer(number_items)[0:number_pick_count]
            other_items = sort_answer(other_items)[0:other_pick_count]
            
            filtered_data.extend(yes_items + no_items + number_items + other_items)
            batch_data.clear()
    
    data_img, _ = create_index(filtered_data)
    print('remaining imgs after filter: {}'.format(len(data_img)))
    count_and_save(filtered_data, base_path)


def create_index(data):
    print('creating index...')
    data_img = {}
    data_q_id = {}
    for item in tqdm(data):
        data_q_id[item['question_id']] = item
        img_questions = data_img.setdefault(item['img_id'], [])
        duplicate = False
        for img_question in img_questions:
            if item['question'] == img_question['question']:
                duplicate = True
                break
        if duplicate is False:
            img_questions.append(item)
    return data_img, data_q_id


def gather_data(base_path):
    q_id_gen = QuestionIdGen()
    
    data = []
    
    datasets = glob.glob('dataset_[0-9]*.json')
    
    for dataset in datasets:
        with open(dataset) as f:
            data.extend(json.load(f)['data'])
    
    for item in data:
        item['question_id'] = q_id_gen.q_id
    
    data = list(filter(lambda item: '[unused1]' not in item['question'] and '[unused1]' not in item['answer'] and item['prob'] > 0 and item['question'] != item['answer'], data))
    random.shuffle(data)
    
    print('raw length: {}'.format(len(data)))
    
    with open(os.path.join(base_path, 'raw_dataset.json'), mode='w') as f:
        json.dump({'data': data}, f, indent=4)
    return data


def sort_answer(data):
    return sorted(data, key=lambda item: item['prob'], reverse=True)


def separate_by_category(data):
    yes_items = list(filter(lambda item: item['type'] == 'yes/no' and item['answer'] == 'yes', data))
    no_items = list(filter(lambda item: item['type'] == 'yes/no' and item['answer'] == 'no', data))
    number_items = list(filter(lambda item: item['type'] == 'number', data))
    other_items = list(filter(lambda item: item['type'] == 'other', data))
    return yes_items, no_items, number_items, other_items


def count_and_save(data, base_path):
    yes_items, no_items, number_items, other_items = separate_by_category(data)

    print('filtered length: {}'.format(len(data)))
    print('yes count: {}'.format(len(yes_items)))
    print('no count: {}'.format(len(no_items)))
    print('number count: {}'.format(len(number_items)))
    print('other count: {}'.format(len(other_items)))
    
    with open(os.path.join(base_path, 'filtered_dataset.json'), 'w') as f:
        json.dump({
            'data': data,
            'count': {
                'yes/no': len(yes_items) + len(no_items),
                'number': len(number_items),
                'other': len(other_items)
            }
        }, f, indent=4)


if __name__ == '__main__':
    # main()
    main2()
