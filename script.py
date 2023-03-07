from vqa_utils.vqaTools.vqa import VQA
from utils import parse_yaml, default_config_path
from data.dataset import CC3MDataset, Tokenizer, build_cc3m_dataset, CaptionProcessor, build_vg_dataset
from torch.utils.data import DataLoader
import torch
import json
from vqa_utils.vqaTools.vqa import VQA
import random
import glob
import tqdm
import torch.distributed as dist
import os


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


def main7():
    # count prob in each question category
    from gather_filter import separate_by_category, separate_number
    with open('./filtered_dataset.json') as f:
        items = json.load(f)
    yes_items, no_items, number_items, other_items = separate_by_category(items['data'])
    common_number_items, long_tail_number_items = separate_number(number_items)
    
    def avg(_data):
        return sum(map(lambda item: item['prob'], _data)) / len(_data)

    def minimum(_data):
        return min(map(lambda item: item['prob'], _data))
    
    print('yes prob - avg: {} - min: {} '.format(avg(yes_items), minimum(yes_items)))
    print('no prob - avg: {} - min: {} '.format(avg(no_items), minimum(no_items)))
    print('common number prob - avg: {} - min: {} '.format(avg(common_number_items), minimum(common_number_items)))
    print('long tail number prob - avg: {} - min: {} '.format(avg(long_tail_number_items), minimum(long_tail_number_items)))
    print('other prob - avg: {} - min: {} '.format(avg(other_items), minimum(other_items)))


def main8():
    # count and categorize question types.
    with open('./dataset/VG/question_answers.json') as f:
        dataset = json.load(f)
    statistics = {}
    tokenizer = Tokenizer().tokenizer
    
    import tqdm
    for data in tqdm.tqdm(dataset):
        for qa in data['qas']:
            words = tokenizer.tokenize(qa['question'])
            item = statistics
            # only count the first 4 words
            for word in words[0:4]:
                item = item.setdefault(word, {
                    'count': 0,
                    'successor': {}
                })
                item['count'] += 1
                item = item['successor']
    
    with open('./custom_dataset/VG/statistics.json', 'w') as f:
        json.dump(statistics, f, indent=4)


def main9():
    # random split train and val dataset in VG
    with open('./dataset/VG/question_answers.json') as f:
        dataset = json.load(f)
    
    ids = []
    for data in dataset:
        if len(data['qas']) > 0:
            ids.append(str(data['id']))
    
    random.shuffle(ids)
    val_size = 2000
    val_id = ids[0:val_size]
    train_id = ids[val_size:]
    
    with open('./custom_dataset/VG/train_id.json', 'w') as f:
        json.dump(train_id, f, indent=4)
    
    with open('./custom_dataset/VG/val_id.json', 'w') as f:
        json.dump(val_id, f, indent=4)
    print('train length: {}'.format(len(train_id)))
    print('val length: {}'.format(len(val_id)))


def main10():
    # show question type statistics
    with open('./custom_dataset/VG/statistics.json') as f:
        statistics = json.load(f)
    
    # the first level
    print('types of the first level: {}'.format(len(statistics)))
    count = {}
    for key, value in statistics.items():
        count[key] = value['count']
    print(count)
    
    # the second level
    for key, value in statistics.items():
        print('level2: {} count'.format(key))
        count = {}
        for _key, _value in value['successor'].items():
            count[_key] = _value['count']
        print(len(count))
        print(sorted(count.items(), key=lambda item: item[1], reverse=True)[0:10])


def main11():
    # check dataset bug
    # from data.dataset import build_vg_dataset, ImageProcessor
    # dataset = build_vg_dataset('train', Tokenizer())
    # print(dataset[4502])
    
    # import os
    # from PIL import Image
    # from torchvision.transforms import *
    # img_path = os.path.join('./dataset/VG/VG_100K', '{}.jpg'.format(2413180))
    # img = Image.open(img_path)
    # # get image size: W, H
    # # img_size = img.size[0], img.size[1]
    # # print(img_size)
    # img = img.convert('RGB')
    # img = ToTensor()(img)
    # print(img.size())
    # img = Resize((224, 224))(img)
    # # print(img.size())
    # import numpy as np
    # Image.fromarray(np.uint8(img.squeeze(0).permute(1, 2, 0) * 255)).save('./test.jpg')

    import cv2
    image = cv2.imread('./test.jpg')
    cv2.rectangle(image, (0, 90), (79, 143), (0, 0, 255), 2)  
    cv2.imwrite('./test2.jpg', image)


def main12():
    # convert generated imgs
    with open('./custom_dataset/CC3M/objects.json') as f:
        items = json.load(f)
    img_ids = list(items.keys())
    selected_imgs = list(map(lambda item: '{}.jpg'.format(item), img_ids))
    with open('./selected_imgs.json', 'w') as f:
        json.dump(selected_imgs, f, indent=4)


def main13():
    # check untrained positional tokens.
    from utils import set_DDP, get_args
    args = get_args()
    set_DDP(args)
    
    # check untrained location token
    import json
    dataset = build_vg_dataset('train', Tokenizer())
    dataloader = DataLoader(dataset, batch_size=64)
    iteration = 50
    
    for _ in range(iteration):
        for _ in tqdm.tqdm(dataloader):
            pass
    token_count = dataset.location_embedding.token_count
    
    untrained_tokens = []
    for i in range(224):
        if str(i) not in token_count:
            untrained_tokens.append(str(i))
    print(untrained_tokens)
    with open('token_count_{}_{}.json'.format(iteration, dist.get_rank()), 'w') as f:
        json.dump(token_count, f, indent=4)


def main14():
    # convert object detection results
    input_dir = '/share/group/datanlpr_ai/zkliu/bu-feat/cc3m_info'
    import numpy as np
    import os
    
    data = {}
    top = 16
    gen_len = 1000000
    
    selected_imgs = []

    for item in tqdm.tqdm(list(glob.glob(os.path.join(input_dir, '*.npz')))[0:gen_len]):
        img_id = os.path.splitext(os.path.split(item)[1])[0]
        result = np.load(item)
        pred_boxes = result['pred_boxes'].tolist()
        scores = result['scores'].tolist()
        if len(pred_boxes) <= 0 or len(scores) <= 0 or len(pred_boxes) != len(scores):
            print('Image {} has no detected object.'.format(img_id))
            continue
        sorted_items = sorted(list(zip(pred_boxes, scores)), key=lambda item: item[1], reverse=True)[0:top]
        data[str(img_id)] = list(zip(*sorted_items))[0]
        selected_imgs.append('{}.jpg'.format(str(img_id)))

    with open('./custom_dataset/CC3M/objects.json', 'w') as f:
        json.dump(data, f, indent=4)

    with open('./selected_imgs.json', 'w') as f:
        json.dump(selected_imgs, f, indent=4)


def main15():
    # seperate large dataset into picture-indexed dataset files.
    with open('filtered_dataset.json') as f:
        data = json.load(f)['data']
    
    data_img, _ = create_index(data)
    print('len data img: {}'.format(len(data_img)))
    for key, value in tqdm.tqdm(data_img.items()):
        with open('/raid/zkliu/CC3M_QA_3M/{}.json'.format(key), 'w') as f:
            json.dump(value, f)


def main16():
    # count yes/no questions in VQAv2 dataset
    tokenizer = Tokenizer().tokenizer
    with open('./dataset/mscoco/v2_mscoco_train2014_annotations.json') as f:
        annotations = json.load(f)['annotations']
    
    with open('./dataset/mscoco/v2_OpenEnded_mscoco_train2014_questions.json') as f:
        questions = json.load(f)['questions']
    
    statistics = {}
    second_statistics = {}
    second_target = 'do'
    
    for question, annotation in tqdm.tqdm(zip(questions, annotations)):
        assert question['question_id'] == annotation['question_id']
        
        if annotation['answer_type'] == 'yes/no':
            words = tokenizer.tokenize(question['question'])
            statistics.setdefault(words[0], 0)
            statistics[words[0]] += 1
            
            if words[0] == second_target:
                second_statistics.setdefault(words[1], 0)
                second_statistics[words[1]] += 1
    
    count = 0
    # check most common question proportion
    for item in ['is', 'does', 'are', 'do']:
        count += statistics.get(item)
    # proportion is 0.920416821466665
    print(statistics)
    print(count / sum(statistics.values()))
    print(second_statistics)


def main17():
    # auto generate yes/no questions using template
    import os
    templates = [
        "Does this picture match the description of '{}'?",
        "Does the sentence '{}' correctly describe the picture?",
        "Do you think this picture matches the description of '{}'?",
        "Is there anything matching the description of '{}' in the picture?",
        "Is the picture consistent with the description of '{}'?",
        "Do the descriptions '{}' and '{}' both match the picture?",
        "Are the descriptions '{}' and '{}' both consistent with this picture?"
    ]
    two_descrption_question = [5, 6]
    
    question_dir = './dataset/cc3m_qa'
    save_dir = './dataset/cc3m_qa_binary'
    gen_len = 5
    question_id = 313773934
    question_items = os.listdir(question_dir)
    print('len imgs:', len(question_items))

    def get_other(original_item):
        _item = original_item
        while _item == original_item:
            _item = random.choice(question_items)
        with open(os.path.join(question_dir, _item)) as f:
            _items = json.load(f)
        return random.choice(_items)['region_description']
    
    def get_this(original_item, count=1):
        with open(os.path.join(question_dir, item)) as f:
            _items = json.load(f)
        if count == 1:
            return random.choice(_items)['region_description']
        else:
            return [random.choice(_items)['region_description'] for _ in range(count)]

    for item in tqdm.tqdm(question_items):
        qas = []
        img_id = item[:-5]
        for _ in range(gen_len):
            index = random.randint(0, len(templates) - 1)
            answer_yes = random.random() < 0.5
            qa = {'img_id': img_id, 'question_id': question_id, 'type': 'binary', 'answer': 'yes' if answer_yes else 'no'}
            question_id += 1
            if index in two_descrption_question:
                if answer_yes:
                    qa['question'] = templates[index].format(*get_this(item, 2))
                elif random.random() < 0.5:
                    captions = [get_this(item), get_other(item)]
                    random.shuffle(captions)
                    qa['question'] = templates[index].format(*captions)
                else:
                    qa['question'] = templates[index].format(get_other(item), get_other(item))
            else:
                if answer_yes:
                    qa['question'] = templates[index].format(get_this(item))
                else:
                    qa['question'] = templates[index].format(get_other(item))
            qas.append(qa)
        with open(os.path.join(save_dir, item), 'w') as f:
            json.dump(qas, f)


def main18():
    # hash match images between VG and GQA
    import imagehash
    from PIL import Image
    import os
    vg_path = './dataset/VG/VG_100K'
    gqa_path = './dataset/GQA/images'
    vg_dict = {}
    gqa_dict = {}
    gqa_to_vg = {}
    
    
    for item in tqdm.tqdm(os.listdir(vg_path)):
        try:
            hash_value = imagehash.phash(Image.open(os.path.join(vg_path, item)))
        except:
            continue
        vg_dict[str(hash_value)] = item[:-4]
    
    for item in tqdm.tqdm(os.listdir(gqa_path)):
        try:
            hash_value = imagehash.phash(Image.open(os.path.join(gqa_path, item)))
        except:
            continue
        gqa_dict[str(hash_value)] = item[:-4]
        if str(hash_value) in vg_dict:
            gqa_to_vg[item[:-4]] = vg_dict[str(hash_value)]
    
    with open('gqa_to_vg.json', 'w') as f:
        json.dump(gqa_to_vg, f)


def main19():
    # merge yes/no questions and original vg questions
    with open('./dataset/GQA/questions1.2/train_balanced_questions.json') as f:
        gqa = json.load(f)
    
    with open('./dataset/GQA/questions1.2/val_balanced_questions.json') as f:
        gqa.update(json.load(f))
    
    data = {}
    for item in tqdm.tqdm(gqa.values()):
        if item['answer'] in ['yes', 'no']:
            data.setdefault(str(item['imageId']), [])
            data[str(item['imageId'])].append({
                'question': item['question'],
                'answer': item['answer'],
                'qa_id': -114514
            })
    
    with open('./dataset/VG/txt_mapper.json') as f:
        vg = json.load(f)
        
    with open('./gqa_to_vg.json') as f:
        mapper = json.load(f)
    
    miss_count = 0
    
    for key, value in tqdm.tqdm(data.items()):
        if key not in mapper:
            continue
        real_key = mapper[key]
        
        if real_key not in vg:
            miss_count += 1
            continue
        vg[real_key].extend(value)
    
    print('miss count: ', miss_count)
    
    with open('./dataset/VG/txt_mapper_with_gqa_binary.json', 'w') as f:
        json.dump(vg, f)


def main20():
    # merge common qa and binary qa data
    items = os.listdir('../CC3M_QA_3M_right_224')
    items2 = os.listdir('../CC3M_QA_3M_binary_right_224_balanced_scale')
    ratio = 0.1
    
    cnt_binary = 0
    cnt_total = 0
    cnt_drop = 0
    items = list(set(items) & set(items2))
    for item in tqdm.tqdm(items):
        with open('../CC3M_QA_3M_right_224/{}'.format(item)) as f:
            qas = json.load(f)
        with open('../CC3M_QA_3M_binary_right_224_balanced_scale/{}'.format(item)) as f:
            binary = json.load(f)

        if len(qas + binary) < 1 or len(qas) < 1:
            continue
        
        random.shuffle(binary)
        temp = len(binary)
        binary = binary[:int(ratio * len(qas))]
        cnt_drop += (temp - len(binary))
        cnt_binary += len(binary)
        cnt_total += (len(binary) + len(qas))
        with open('../CC3M_QA_3M_right_224_with_right_balanced_scaled_binary/{}'.format(item), 'w') as f:
            json.dump(qas + binary, f)
    print('binary: {} - total: {} - ratio: {} - drop: {}'.format(cnt_binary, cnt_total, cnt_binary / cnt_total, cnt_drop))


def main21():
    # create CC3M caption + qa dataset in VALOR format
    cc3m_qa_path = '../CC3M_QA_3M_right_224_with_right_balanced_scaled_binary'
    dataset_path = '../CC3M_cap_qa'
    ids = [item[:-5] for item in os.listdir(cc3m_qa_path)]
    with open('./dataset/cc3m/txt_mapper.json') as f:
        txt_mapper = json.load(f)
    
    for _id in tqdm.tqdm(ids):
        with open(os.path.join(cc3m_qa_path, '{}.json'.format(_id))) as f:
            qas = json.load(f)
        caption = txt_mapper[_id]
        anno = {}
        anno['video_id'] = _id
        anno['desc'] = caption
        anno['question'], anno['answer'] = tuple(zip(*list(map(lambda item: (item['question'], item['answer']), qas))))
        
        with open(os.path.join(dataset_path, '{}.json'.format(_id)), 'w') as f:
            json.dump(anno, f)


def main22():
    # count the distribution of CC3M-QA question types
    cc3m_qa_path = '../CC3M_QA_3M_right_224_with_right_balanced_scaled_binary'
    imgs = os.listdir(cc3m_qa_path)
    statistics = {}
    tokenizer = Tokenizer().tokenizer
    
    import tqdm
    for img in tqdm.tqdm(imgs):
        with open(os.path.join(cc3m_qa_path, img)) as f:
            data = json.load(f)
        for qa in data:
            words = tokenizer.tokenize(qa['question'])
            item = statistics
            # only count the first 4 words
            for word in words[0:4]:
                item = item.setdefault(word, {
                    'count': 0,
                    'successor': {}
                })
                item['count'] += 1
                item = item['successor']
    
    with open('./custom_dataset/CC3M/statistics.json', 'w') as f:
        json.dump(statistics, f, indent=4)


def main23():
    # extract CC3M dense caption
    import tqdm
    cc3m_qa_path = '../CC3M_QA_3M'
    imgs = os.listdir(cc3m_qa_path)
    max_ = -1
    min_ = 1000
    
    for img in tqdm.tqdm(imgs):
        with open(os.path.join(cc3m_qa_path, img)) as f:
            data = json.load(f)
        video_id = img[:-5]
        region = []
        region_desc = []
        for qa in data:
            if qa['object'] in region and qa['region_description'] in region_desc:
                continue
            region.append(qa['object'])
            region_desc.append(qa['region_description'])
            
            for value in qa['object']:
                if value > max_:
                    max_ = value
                if value < min_:
                    min_ = value
        
        with open(os.path.join('../CC3M_region_desc', img), 'w') as f:
            json.dump({'video_id': video_id, 'region': region, 'region_desc': region_desc}, f)
    
    print('max: {}, min: {}'.format(max_, min_))


def create_index(data):
    data_img = {}
    data_q_id = {}
    for item in tqdm.tqdm(data):
        data_q_id[item['question_id']] = item
        img_questions = data_img.setdefault(item['img_id'], [])
        img_questions.append(item)
    return data_img, data_q_id


if __name__ == '__main__':
    main22()
