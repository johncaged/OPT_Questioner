from model.model import BaseQuestioner, ModelWrapper, QuestionerWithAnswer, QuestionerWithCaption, TextSampler, TopKSampler, BeamSampler, TwoStageSampler
from data.dataset import Tokenizer, build_dataset, CaptionProcessor, build_cc3m_dataset, CC3MDataset, build_dataloader
from torch.utils.data import DataLoader
from utils import ToCuda, QuestionIdGen
import torch
import json
import time
import torch.distributed as dist
import warnings
import random
import os
import subprocess
import argparse

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--use_slurm', type=str2bool, default=False)
    parser.add_argument('--master_address', type=str, default=None)
    parser.add_argument('--master_port', type=str, default=None)
    # parser.add_argument('--fp16', action='store_true')
    return parser.parse_args()


def setup_DDP(backend='nccl', address=None, port=None, verbose=False):
    '''Initialize slurm distributed training environment.
    '''
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    # specify master address
    if address is not None:
        os.environ['MASTER_ADDR'] = address
    elif 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)

    # The following code is the same as the setup_DDP() code in single-machine-and-multi-GPU-DistributedDataParallel-launch.py
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # If the OS is Windows or macOS, use gloo instead of nccl
    dist.init_process_group(backend=backend)
    # set distributed device
    device = torch.device('cuda:{}'.format(local_rank))
    if verbose:
        print('Using device: {}'.format(device))
        print(f'local rank: {local_rank}, global rank: {rank}, world size: {world_size}')
    return rank, local_rank, world_size, device


def main():
    q_id_gen = QuestionIdGen()
    
    args = get_args()
    
    # parallel
    if args.use_slurm is False:
        dist.init_process_group('nccl')
        rank = dist.get_rank()
    else:
        print('use slurm to run distributed pytorch program.')
        _, rank, _, _ = setup_DDP(address=args.master_address, port=args.master_port)
    
    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    with torch.no_grad():
        # with open('images.json') as f:
        #     all_images = json.load(f)
        # tokenizer
        tokenizer = Tokenizer()
        # build and load model
        model = BaseQuestioner(tokenizer, QuestionerWithCaption(), auto_regressive=True)
        # model = BaseQuestioner(tokenizer, QuestionerWithAnswer(), auto_regressive=True)
        model = ModelWrapper(model)
        # checkpoint = torch.load('./log/test/checkpoint/checkpoint.pt', map_location='cpu')
        checkpoint = torch.load('./log/answer_type_224_512_train_train2/checkpoint/best.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.eval()
        del checkpoint
        model: TextSampler = ToCuda(TwoStageSampler(model.module, base_k=10, question_answer_sep_token=tokenizer.question_answer_sep_token, answer_type_mask_token=tokenizer.answer_type_mask_token))
        # model: TextSampler = ToCuda(TopKSampler(model.module, end_token=tokenizer.question_answer_sep_token, k=5))
        # answer_model: TextSampler = ToCuda(TopKSampler(model.module, k=1))
        # answer_model: TextSampler = ToCuda(BeamSampler(model.module))
        # val_dataset = DataLoader(build_dataset('caption_only', 'val', tokenizer, 'all'), batch_size=1, shuffle=True)
        # val_dataset = DataLoader(build_dataset('answer', 'val', tokenizer, 'all'), batch_size=1, shuffle=True)
        # val_dataset = DataLoader(build_dataset('caption', 'val', tokenizer, 'all', text_encoder=model.module.clip), batch_size=1, shuffle=True)
        # batch_size = 2048
        batch_size = 256 * dist.get_world_size()
        # val_dataset = DataLoader(build_cc3m_dataset(tokenizer), batch_size=batch_size, shuffle=True, collate_fn=CC3MDataset.collate_fn)
        cc3m = build_cc3m_dataset(tokenizer)
        gen_imgs = []
        if os.path.exists('generated_imgs.json'):
            with open('generated_imgs.json') as f:
                gen_imgs = json.load(f)
        print('raw dataset size: {}'.format(len(cc3m)))
        print('generated img size: {}'.format(len(gen_imgs)))
        
        cc3m.img_names = list(set(cc3m.img_names).difference(set(gen_imgs)))
        print('dataset left: {}'.format(len(cc3m)))
        
        val_dataset, _ = build_dataloader(cc3m, batch_size=batch_size, collate_fn=CC3MDataset.collate_fn)
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
        
        yes_no_per_img = 4
        number_per_img = 4
        other_per_img = 4
        
        types = [
            *[{'answer_type': 'yes/no'} for _ in range(yes_no_per_img)],
            *[{'answer_type': 'number'} for _ in range(number_per_img)],
            *[{'answer_type': 'other'} for _ in range(other_per_img)]
        ]
        
        no_prob = 0.75
        zero_prob = 1e-4
        
        repeat_sample = len(types)
        
        # max_iter = 100000 // batch_size
        
        for i, batch in enumerate(val_dataset):
            start_time = time.time()
            
            tip = batch[0]['tips']
            img_ids = batch[1]
            
            data_to_select = {}
            # target = torch.flatten(batch[1], start_dim=0, end_dim=1)
            
            # img = []
            # for item in zip(*batch[2]):
            #     img.extend(list(item))
            # if img[0] in all_images['images']:
            # if img[0] not in all_images['images']:
                # max_iter += 1
                # continue
            
            # append_to_file('result.txt', 'Choose Image: {0}\n\n'.format(img_ids[0]))
            
            for j in range(repeat_sample):
                start_sample = time.time()
                
                answer_type = types[j]['answer_type']
                if answer_type == 'yes/no':
                    answer_type = 'no' if random.random() < no_prob else 'yes'
                elif answer_type == 'number':
                    answer_type = 'zero' if random.random() < zero_prob else 'number'
                
                batch[0]['tips'] = CaptionProcessor.attach_task_prompt(tip, answer_type, False, tokenizer)
                
                # append_to_file('result.txt', 'Answer Type: {0}, Similar: {1}\n'.format(types[j]['answer_type'], types[j]['similar']))
                
                prediction, probability = model(batch[0])
                
                # print(probability)
                # _question_prediction = detach(question_prediction)
                # _question_prediction = [item[0:item.index(tokenizer.question_answer_sep_token) + 1] for item in _question_prediction]
                # max_length = max(map(lambda item: len(item), _question_prediction))
                # _question_prediction = [([0] * (max_length - len(item))) + item for item in _question_prediction]
                
                # batch[0]['tips'] = torch.cat([
                #     batch[0]['tips'],
                #     torch.tensor(_question_prediction).long()
                    # torch.where(question_prediction != tokenizer.question_answer_sep_token, question_prediction, 0),
                    # ToCuda(torch.tensor([tokenizer.question_answer_sep_token]).unsqueeze(0).expand(question_prediction.shape[0], -1).long())
                # ], dim=1)
                
                # batch[0]['questions'] = ToCuda(torch.tensor(_question_prediction).long())
                
                # print(convert_items(detach(batch[0]['tips']), tokenizer, tokenizer.question_answer_sep_token))
                
                # answer_prediction = answer_model(batch[0])
                
                # del batch[0]['questions']

                for answer, question, prob, img_id in zip(convert_items(detach(prediction), tokenizer, start_token=tokenizer.question_answer_sep_token),
                                                          convert_items(detach(prediction), tokenizer, end_token=tokenizer.question_answer_sep_token),
                                                          # convert_items(detach(question_prediction), tokenizer),
                                                          detach(probability),
                                                          img_ids):
                    img_data = data_to_select.setdefault(img_id, {
                        'yes/no': [],
                        'number': [],
                        'other': [],
                        'all': []
                    })
                    
                    _data = {
                        'question': question,
                        'answer': answer,
                        'prob': prob,
                        'type': types[j]['answer_type']
                    }
                    
                    img_data[types[j]['answer_type']].append(_data)
                    img_data['all'].append(_data)
                    # append_to_file('result.txt', ('Caption: {0} - Question: {1} - Answer: {2} - Probability: {3}\n'.format(
                    #     caption,
                    #     question,
                    #     answer,
                    #     prob
                    # )))
                # append_to_file('result.txt', '\n')
                end_sample = time.time()
                
                if dist.get_rank() == 0:
                    print('sample {} - time {}s'.format(j, end_sample - start_sample))
            # items = map(lambda x: int(x), list(set(img)))
            # all_images['images'].extend(items)
            for img_id, img_data in data_to_select.items():
                selected_data = []
                selected_data.extend(img_data['yes/no'])
                selected_data.extend(img_data['number'])
                selected_data.extend(img_data['other'])
                for _data in selected_data:
                    _data['img_id'] = img_id
                    _data['question_id'] = q_id_gen.q_id
                data['data'].extend(selected_data)
            
            with open('dataset_{}.json'.format(dist.get_rank()), 'w') as f:
                json.dump(data, f, indent=4)
            # with open('images.json', 'w') as f:
            #     json.dump(all_images, f, indent=4)

            # if i >= max_iter - 1:
            #     break
            
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


def str2bool(b):
    if b.lower() in ["false"]:
        return False
    elif b.lower() in ["true"]:
        return True
    elif b is None:
        return None
    else:
        raise Exception("Invalid Bool Value")


if __name__ == '__main__':
    main()
