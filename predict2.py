from model.model import BaseQuestioner, ModelWrapper, QuestionerWithAnswer, QuestionerWithCaption, TextSampler, TopKSampler, BeamSampler, TwoStageSampler, MultiStageAdapter
from data.dataset import Tokenizer, build_coco_dataset, CaptionProcessor, build_cc3m_dataset, CC3MDataset, build_dataloader
from torch.utils.data import DataLoader
from utils import ToCuda, QuestionIdGen
import torch
import json
import time
import torch.distributed as dist
import warnings
import random
import os
from utils import get_args, set_DDP

warnings.filterwarnings("ignore")


@torch.no_grad()
def main():
    q_id_gen = QuestionIdGen()
    
    args = get_args()
    set_DDP(args)
    
    tokenizer = Tokenizer()
    # build and load model
    model = BaseQuestioner(tokenizer, MultiStageAdapter(), auto_regressive=True)
    model = ModelWrapper(model)
    checkpoint = torch.load('./log/vg_224_512/checkpoint/checkpoint_39.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    del checkpoint
    caption_model: TextSampler = ToCuda(TopKSampler(model.module, k=3))
    model: TextSampler = ToCuda(TwoStageSampler(model.module, base_k=10, question_answer_sep_token=tokenizer.question_answer_sep_token, answer_type_mask_token=tokenizer.answer_type_mask_token))
    
    batch_size = 256 * dist.get_world_size()
    
    cc3m = build_cc3m_dataset(tokenizer)
    gen_imgs = []
    if os.path.exists('generated_imgs.json'):
        with open('generated_imgs.json') as f:
            gen_imgs = json.load(f)
    print('raw dataset size: {}'.format(len(cc3m)))
    print('generated img size: {}'.format(len(gen_imgs)))
    
    cc3m.img_names = list(set(cc3m.img_names).difference(set(gen_imgs)))
    
    if args.gen_selected_imgs is True:
        with open('selected_imgs.json') as f:
            cc3m.img_names = json.load(f)
    print('dataset left: {}'.format(len(cc3m)))
    
    val_dataset, _ = build_dataloader(cc3m, batch_size=batch_size, collate_fn=CC3MDataset.collate_fn, shuffle=False)
    print('step left: {}'.format(len(val_dataset)))
    
    detach = lambda x: x.detach().cpu().tolist()
    
    if os.path.exists('dataset_{}.json'.format(dist.get_rank())):
        with open('dataset_{}.json'.format(dist.get_rank())) as f:
            data = json.load(f)
    else:
        data = {
            'data': []
        }
    
    print('generated data length at rank {}: {}'.format(dist.get_rank(), len(data['data'])))
    
    types = [
        *[{'answer_type': 'what'} for _ in range(1)],
        *[{'answer_type': 'how'} for _ in range(1)],
        *[{'answer_type': 'where'} for _ in range(1)],
        *[{'answer_type': 'who'} for _ in range(1)],
        *[{'answer_type': 'why'} for _ in range(1)],
        *[{'answer_type': 'when'} for _ in range(1)]
    ]
    
    repeat_sample = len(types)
    object_sample = 3
    
    # max_iter = 100000 // batch_size
    
    def process_tip(*tip_items):
        _tip = tokenizer.concat_tokens(tip_items, tokenizer.task_prompt_sep_token)
        _tip = tokenizer.get_padded_tokens(_tip).unsqueeze(0)
        return _tip

    def convert_object(_object):
        return [int(_item[7:-1]) - 100 for _item in _object]
    
    for i, batch in enumerate(val_dataset):
        start_time = time.time()
        
        img_ids = batch[1]
        
        data_to_select = {}
        
        for k in range(object_sample):
            objects = batch[0]['objects']
            sampled_objects = [random.choice(_objects) for _objects in objects]
            # generate region captions
            processed_sampled_objects = [process_tip(_object) for _object in sampled_objects]
            batch[0]['caption_tips'] = torch.cat(processed_sampled_objects, dim=0)
            caption_model.module._forward = 'caption'
            prediction = caption_model(batch[0])
            captions = convert_items(detach(prediction), tokenizer)
            
            model.module._forward = 'question'
            processed_tip = torch.cat([process_tip(_object, region_caption) for _object, region_caption in zip(sampled_objects, captions)], dim=0)
        
            for j in range(repeat_sample):
                start_sample = time.time()
                
                answer_type = types[j]['answer_type']
                
                batch[0]['tips'] = CaptionProcessor.attach_task_prompt(processed_tip, answer_type, tokenizer)
                
                prediction, probability = model(batch[0])

                for answer, question, prob, img_id, _object, _caption in zip(convert_items(detach(prediction), tokenizer, start_token=tokenizer.question_answer_sep_token),
                                                            convert_items(detach(prediction), tokenizer, end_token=tokenizer.question_answer_sep_token),
                                                            # convert_items(detach(question_prediction), tokenizer),
                                                            detach(probability),
                                                            img_ids,
                                                            sampled_objects,
                                                            captions):
                    img_data = data_to_select.setdefault(img_id, [])
                    _data = {
                        'question': question,
                        'answer': answer,
                        'prob': prob,
                        'type': types[j]['answer_type'],
                        'object': convert_object(_object),
                        'region_description': _caption
                    }
                    
                    img_data.append(_data)
                end_sample = time.time()
                
                if dist.get_rank() == 0:
                    print('sample {} - time {}s'.format(j, end_sample - start_sample))
        
        for img_id, img_data in data_to_select.items():
            for _data in img_data:
                _data['img_id'] = img_id
                _data['question_id'] = q_id_gen.q_id
            data['data'].extend(img_data)
        
        with open('dataset_{}.json'.format(dist.get_rank()), 'w') as f:
            json.dump(data, f, indent=4)
        
        end_time = time.time()
        print('Rank: {} - Iter: {} - Step Time: {} - ETA: {}s'.format(dist.get_rank(), i, end_time - start_time, (end_time - start_time) * (len(val_dataset) - i - 1)))


def append_to_file(path, content):
    with open(path, 'a') as f:
        f.write(content)


def convert_items(items, tokenizer: Tokenizer, end_token=102, start_token=101):
    def find(_list, item):
        try:
            return _list.index(item)
        except Exception:
            return -1
    
    results = []
    for item in items:
        results.append(' '.join(tokenizer.tokenizer.convert_ids_to_tokens(item[find(item, start_token) + 1:find(item, end_token)])).replace(' ##', ''))
    return results


if __name__ == '__main__':
    main()
