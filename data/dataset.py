from model.bert_tokenizer import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from utils import default_config_path, parse_yaml, ToCuda
from vqa_utils.vqaTools.vqa import VQA
import os
import random
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import *
from torch_lib.util import NOTHING
import json
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from model.clip_tokenizer import tokenize
import torch.nn as nn
import copy
from model.clip import CLIP
import collections
import re


'''------------------------------------------dataset utils.------------------------------------------'''
class Tokenizer:
    
    def __init__(self, max_len: int = 30, config_path=default_config_path):
        super().__init__()
        # max text length
        self.max_len = max_len
        config = parse_yaml(config_path)
        self.tokenizer = BertTokenizer(os.path.join(config['initial_path'], config['bert_vocab_path']))
        
        # get cls token and sep token
        self.cls_token = self.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.sep_token = self.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
        # mask token
        self.mask_token = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        # task prompt sep token
        self.task_prompt_sep_token = self.tokenizer.convert_tokens_to_ids(['[unused0]'])[0]
        # question answer sep token
        self.question_answer_sep_token = self.tokenizer.convert_tokens_to_ids(['[unused1]'])[0]
        # answer type mask token
        self.answer_type_mask_token = self.tokenizer.convert_tokens_to_ids(['[unused2]'])[0]
        assert self.cls_token == 101 and self.sep_token == 102, 'The cls token or the sep token does not match the correct id.'

    def tokenize(self, text: str, max_len: int = None):
        return self.get_padded_tokens(self.tokenize_without_padding(text), max_len)

    def tokenize_without_padding(self, text):
        tokenized_text = self.tokenizer.tokenize(text)
        tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return tokens
    
    def get_padded_tokens(self, tokens, max_len: int = None):
        max_len = max_len if max_len is not None else self.max_len
        tokens = tokens[:max_len]
        tokens = [self.cls_token] + tokens + [self.sep_token]
        
        # parse to torch tensor
        tokens = torch.tensor(tokens, dtype=torch.long)
        output = torch.zeros(max_len + 2, dtype=torch.long)
        output[:len(tokens)] = tokens
        return output

    def concat_tokens(self, tokens, sep):
        def encode(_item):
            return self.tokenize_without_padding(_item) if re.fullmatch('\[unused.*\]', _item) is None else self.tokenizer.convert_tokens_to_ids([_item])
        
        _tokens = []
        for token in tokens:
            if isinstance(token, (list, tuple)):
                temp = []
                for t in token:
                    temp.extend(encode(t))
            else:
                temp = encode(token)
            _tokens.append(temp)
        result = []
        for token in _tokens:
            result.extend(token)
            result.append(sep)
        result.pop(-1)
        return result


class ImageProcessor:
    
    def __init__(self, config_path=default_config_path):
        super().__init__()
        self.mean = [0.48145466, 0.4578275, 0.40821073] 
        self.std  = [0.26862954, 0.26130258, 0.27577711]
        
        config = parse_yaml(config_path)
        self.resolution = config['video_resolution']
        self.transforms = NOTHING
    
    def process(self, img):
        img = img.convert('RGB')
        img = ToTensor()(img)
        return self.transforms(img).unsqueeze(0)


class TrainImageProcessor(ImageProcessor):
    
    def __init__(self):
        super().__init__()
        self.transforms = Compose([Resize((self.resolution, self.resolution)),
                                #    RandomResizedCrop(self.resolution, [0.8,1.0],[1.0,1.0]),
                                #    RandomHorizontalFlip(),
                                   Normalize(self.mean, self.std)])


class ValImageProcessor(ImageProcessor):
    
    def __init__(self):
        super().__init__()
        self.transforms = Compose([Resize((self.resolution, self.resolution)),
                                #    CenterCrop(self.resolution),
                                   Normalize(self.mean, self.std)])


class RawResolutionImageProcessor(ImageProcessor):
    
    def __init__(self):
        super().__init__()
        self.transforms = Compose([Normalize(self.mean, self.std)])


class LocationEmbedding:
    
    def __init__(
        self,
        start_from: int = 100
    ):
        self.token_count = {}
        self.start_from = start_from

    def __call__(self, resolution, img_size, region):
        # embed location
        x1, y1, x2, y2 = self.transfer_location(resolution, img_size, region)
        
        def count(_key):
            self.token_count.setdefault(str(_key), 0)
            self.token_count[str(_key)] = self.token_count[str(_key)] + 1
            return count

        count(x1)(y1)(x2)(y2)
        return self.get_location_tokens(x1, y1, x2, y2)

    def transfer_location(self, resolution, img_size, region):
        if region is None:
            # full image
            x1, y1, x2, y2 = 0, 0, resolution - 1, resolution - 1
        else:
            # image region
            x1 = min(round(region[0] * resolution / img_size[0]), resolution - 1)
            y1 = min(round(region[1] * resolution / img_size[1]), resolution - 1)
            x2 = min(round(region[2] * resolution / img_size[0]), resolution - 1)
            y2 = min(round(region[3] * resolution / img_size[1]), resolution - 1)
        return x1, y1, x2, y2

    def get_location_tokens(self, *values):
        return ['[unused{}]'.format(value + self.start_from) for value in values]


'''
------------------------------------------The COCO Dataset.------------------------------------------
'''
class TextProcessor:
    
    def __init__(
        self,
        question_path,
        annotation_path,
        tokenizer: Tokenizer,
        mode: str = 'once',
        confident: bool = False
    ):
        super().__init__()
        self.vqa = VQA(annotation_path, question_path)
        self.confident = confident
        self.tokenizer = tokenizer
        
        assert mode in ['once', 'all']
        # WARNING: If the mode is set to 'all', the batch size may vary during the iteration process.
        self.mode = mode
    
    def process(self, image_id, image_name):
        pass
    
    def get_tokenized_text_pair(self, text1, text2):
        """Get a single tokenized text pair.
        """
        return self.tokenizer.tokenize(text1).unsqueeze(0), self.tokenizer.tokenize(text2).unsqueeze(0)


class QuestionAnswerProcessor(TextProcessor):
    """Get question-answer pair for question generation.
    """
    
    def __init__(self, multiple_choice_answer: bool = True, quesTypes: list = [], ansTypes: list = [], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multiple_choice_answer = multiple_choice_answer
        self.quesTypes = quesTypes
        self.ansTypes = ansTypes

    def process(self, image_id, image_name):
        # get questions
        question_ids = self.vqa.getQuesIds(imgIds=image_id, quesTypes=self.quesTypes, ansTypes=self.ansTypes)
        
        questions, answers = None, None
        # image -> n * questions -> n * m answers
        if self.mode == 'once':
            # random select 1 question from n questions
            q_id = random.choice(question_ids)
            questions, answers = self.get_multiple_pairs([q_id])
        elif self.mode == 'all':
            questions, answers = self.get_multiple_pairs(question_ids)
        return questions, answers
    
    def get_multiple_pairs(self, q_ids):
        questions = []
        answers = []
        
        for q_id in q_ids:
            temp = self.vqa.loadQA(q_id)[0]
            # choose the most frequent answer
            if self.multiple_choice_answer is True:
                question, answer = self.get_single_pair(self.vqa.qqa[q_id]['question'], temp['multiple_choice_answer'])
            else:
                question, answer = self.get_single_pair(self.vqa.qqa[q_id]['question'], temp['answers'])
            questions.append(question)
            answers.append(answer)
        return questions, answers
    
    def get_single_pair(self, question, answers):
        # choose the most frequent answer
        if self.multiple_choice_answer is True:
            answer = {'answer': answers}
        else:
            if self.confident:
                temp = filter(lambda answer: answer.answer_confidence == 'yes', answers)
                if len(temp) >= 1:
                    answers = temp
            answer = random.choice(answers)
        return self.get_tokenized_text_pair(question, answer['answer'])


class QuestionCaptionProcessor(TextProcessor):
    """Get question-caption pair for question generation.
    """
    
    def __init__(self, caption_path: str, text_encoder, k=0.4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(caption_path) as f:
            self.captions = json.load(f)
        # self.text_encoder = CLIPText(text_encoder)
        # self.k = k
        # self.probability = 0.5
    
    def process(self, image_id, image_name):
        # get captions
        img_captions = self.captions[image_name]
        # get questions
        question_ids = self.vqa.getQuesIds(imgIds=image_id)
        
        targets, tips = None, None
        
        # image -> n * questions & m captions
        if self.mode == 'once':
            # random select 1 question from n questions
            img_caption = random.choice(img_captions)
            targets, tips = self.get_multiple_pairs(question_ids, [img_caption])
        elif self.mode == 'all':
            targets, tips = self.get_multiple_pairs(question_ids, img_captions)
        return targets, tips

    def get_multiple_pairs(self, img_question_ids, img_captions):
        """Get multiple question-caption pairs through an array of ids.
        """
        # question + [unused1] + answer
        targets = []
        # answer_type + [unused0] + similar + [unused0] + caption
        tips = []
        
        # img_questions = list(map(lambda q_id: self.vqa.qqa[q_id]['question'], img_question_ids))
        
        # choose one caption for every single question.
        for img_caption in img_captions:
            # choose through clip cosine similarity.
            # question, question_index, similar = self.choose(img_questions, img_caption)
            
            # q_id = img_question_ids[question_index]
            q_id = random.choice(img_question_ids)
            question = self.vqa.qqa[q_id]['question']
            item = self.vqa.loadQA(q_id)[0]
            answer = item['multiple_choice_answer']
            answer_type = item['answer_type']
            if answer_type == 'yes/no':
                random_answer_type = 'yes' if random.random() < 0.5 else 'no'
                answer_type = 'yes' if answer == 'yes' else 'no' if answer == 'no' else random_answer_type
            elif answer_type == 'number' and answer == '0':
                answer_type = 'zero'
            # similar_token = 'similar' if similar else 'different'
            
            # tip = self.tokenizer.concat_tokens([answer_type, similar_token, img_caption], self.tokenizer.task_prompt_sep_token)
            tip = self.tokenizer.concat_tokens([answer_type, img_caption], self.tokenizer.task_prompt_sep_token)
            tip = self.tokenizer.get_padded_tokens(tip).unsqueeze(0)
            target = self.tokenizer.concat_tokens([question, answer], self.tokenizer.question_answer_sep_token)
            target = self.tokenizer.get_padded_tokens(target).unsqueeze(0)
            targets.append(target)
            tips.append(tip)
        return targets, tips
    
    # def choose(self, questions, caption):
    #     caption_token = tokenize(caption)
    #     question_tokens = tokenize(questions)
    #     # whether to choose the most similar question to the caption or not.
    #     similar = random.random() < self.probability
        
    #     class TrainingMode:
    #         def __init__(self, model):
    #             self.model = model
    #             self.training = model.training
            
    #         def __enter__(self):
    #             self.model.eval()
            
    #         def __exit__(self, *args):
    #             self.model.train(self.training)
        
    #     with torch.no_grad():
    #         with TrainingMode(self.text_encoder):
    #             caption_feature, text = self.text_encoder.encode_text(caption_token, casual=False)
    #             caption_feature = caption_feature[torch.arange(caption_feature.shape[0]), torch.argmax(text, dim=-1)]
    #             caption_feature = caption_feature / caption_feature.norm(dim=1, keepdim=True)
                
    #             question_features, text = self.text_encoder.encode_text(question_tokens, casual=False)
    #             question_features = question_features[torch.arange(question_features.shape[0]), torch.argmax(text, dim=-1)]
    #             question_features = question_features / question_features.norm(dim=1, keepdim=True)
                
    #             similarity = torch.mm(caption_feature, question_features.permute(1, 0))
    #             sorted_similarity, indices = torch.sort(similarity, descending=similar)
    #             sim_rank = indices.tolist()[0]
    #             del similarity, sorted_similarity, question_features, caption_feature, text, caption_token, question_tokens
    #     length = min(max(int(((1 - self.k) if similar else self.k) * len(sim_rank)), 2), len(sim_rank))
    #     question_index = random.choice(sim_rank[0:length])
    #     return questions[question_index], question_index, similar


class VQADataset(Dataset):
    
    def __init__(
        self,
        image_processor: ImageProcessor,
        text_processor: TextProcessor,
        image_path: str,
        image_prefix: str,
        id_path: str
    ):
        super().__init__()
        self.image_path = image_path
        self.image_prefix = image_prefix
        self.image_processor = image_processor
        self.text_processor = text_processor
        with open(id_path) as f:
            self.ids = json.load(f)

    def __getitem__(self, index):
        # get image
        image_id = self.ids[index]
        image_name = f'{self.image_prefix}{image_id:012}'
        image_path = os.path.join(self.image_path, f'{image_name}.jpg')
        img = Image.open(image_path)
        img = self.image_processor.process(img).unsqueeze(0)
        
        # shape -> num_items, length of a sentence
        targets, tips = self.text_processor.process(image_id, image_name)
        # generate torch tensor
        if targets is not None and tips is not None:
            assert len(targets) == len(tips), 'The length of questions and tips should be the same.'
            targets = torch.cat(targets, dim=0)
            tips = torch.cat(tips, dim=0)
        # batch, time, channel, height, width
        imgs = img.repeat(tips.size()[0], *([1] * 4))
        # WARNING: the two targets variable are referring to the same memory address.
        return {'imgs': imgs, 'tips': tips, 'targets': targets}, targets, [str(image_id)] * tips.size()[0]

    def __len__(self):
        return len(self.ids)


def build_coco_dataset(
    dataset_type: str,
    mode: str,
    tokenizer: Tokenizer,
    text_mode: str = 'once',
    confident: bool = False,
    multiple_choice_answer: bool = True,
    quesTypes: list = [],
    ansTypes: list = [],
    text_encoder = None,
    k: float = 0.4
):
    img_processor_dict = {
        'train': TrainImageProcessor,
        'train2': TrainImageProcessor,
        'val': ValImageProcessor
    }
    config = parse_yaml(default_config_path)
    # mode: train / val
    items = config[mode]
    # text processor
    text_processor = QuestionAnswerProcessor(multiple_choice_answer, quesTypes, ansTypes, items['question'], items['annotation'], tokenizer, text_mode, confident) if dataset_type == 'answer' else \
        QuestionCaptionProcessor(items['caption'], text_encoder, k, items['question'], items['annotation'], tokenizer, text_mode) if dataset_type == 'caption' else \
        None
    return VQADataset(img_processor_dict[mode](), text_processor, items['image'], items['image_prefix'], items['id_path'])


class CLIPText(nn.Module):
    
    def __init__(self, text_encoder):
        super().__init__()
        module_copy = lambda m: copy.deepcopy(m.cpu() if isinstance(m, nn.Module) else m.cpu().detach())
        
        self.token_embedding = module_copy(text_encoder.token_embedding)
        self.positional_embedding = module_copy(text_encoder.positional_embedding)
        self.prompt_embedding = module_copy(text_encoder.prompt_embedding)
        self.transformer = module_copy(text_encoder.transformer)
        self.ln_final = module_copy(text_encoder.ln_final)
        
        self.text_encoder = text_encoder
        self.transformer_heads = text_encoder.transformer_heads
    
    @property
    def dtype(self):
        return self.text_encoder.dtype
    
    def encode_text(self, txt_tokens, casual=False):
        return CLIP.encode_text(self, txt_tokens, casual=casual)


'''
------------------------------------------The CC3M Dataset.------------------------------------------
'''
class CaptionProcessor:
    
    def __init__(self, caption_path: str, tokenizer: Tokenizer, mode: str = 'once'):
        super().__init__()
        with open(caption_path) as f:
            self.captions = json.load(f)
        self.tokenizer = tokenizer
        self.mode = mode
    
    def process(self, image_id):
        # get captions
        img_captions = self.captions[image_id]
        img_captions = img_captions if isinstance(img_captions, (list, tuple)) else [img_captions]
        
        parse_text = lambda item: self.tokenizer.tokenize(item).unsqueeze(0)
        
        if self.mode == 'once':
            caption = random.choice(img_captions)
            return [parse_text(caption)]
        elif self.mode == 'all':
            img_captions = [parse_text(caption) for caption in img_captions]
            return img_captions

    @staticmethod
    def attach_task_prompt(item, prompt, tokenizer: Tokenizer):
        batch_size = item.shape[0]
        torch_convert = lambda item: torch.tensor(item).unsqueeze(0).expand(batch_size, -1).long()
        
        prompt_token = torch_convert(tokenizer.tokenize_without_padding(prompt))
        return torch.cat([
            item[:, 0:1],
            prompt_token,
            torch_convert([tokenizer.task_prompt_sep_token]),
            item[:, 1:]
        ], dim=1)


class CC3MDataset(Dataset):
    
    def __init__(
        self,
        tokenizer: Tokenizer,
        image_processor: ImageProcessor,
        text_processor: CaptionProcessor,
        location_embedding: LocationEmbedding,
        return_objects: bool,
        config_path=default_config_path
    ):
        super().__init__()
        config = parse_yaml(config_path)
        self.img_names = []
        with open(config['cc3m']['train_meta_path']) as f:
            self.img_names = list(map(lambda item: item.strip().replace('\n', '').replace('\r', ''), f.readlines()))
        
        if return_objects is True:
            with open(config['cc3m']['objects_path']) as f:
                self.object_mapper = json.load(f)
        
        self.return_objects = return_objects
        self.resolution = config['video_resolution']
        self.img_path = config['cc3m']['img_path']
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.text_processor = text_processor
        self.location_embedding = location_embedding
    
    def __getitem__(self, index):
        img_name = self.img_names[index]
        
        img_path = os.path.join(self.img_path, img_name)
        img = Image.open(img_path)
        img_size = img.size[0], img.size[1]
        img = self.image_processor.process(img).unsqueeze(0)
        
        img_id = os.path.splitext(os.path.split(img_name)[1])[0]
        captions = self.text_processor.process(img_id)
        captions = torch.cat(captions, dim=0)
        
        if self.return_objects is True:
            _objects = self.object_mapper[str(img_id)]
            objects = [self.location_embedding(self.resolution, img_size, _object) for _object in _objects]
        else:
            objects = []
        
        # batch, time, channel, height, width
        imgs = img.repeat(captions.size()[0], *([1] * 4))
        return {'imgs': imgs, 'tips': captions, 'objects': objects}, str(img_id)
    
    def __len__(self):
        return len(self.img_names)

    @staticmethod
    def collate_fn(batch):
        imgs = list(map(lambda item: item[0]['imgs'], batch))
        tips = list(map(lambda item: item[0]['tips'], batch))
        objects = list(map(lambda item: item[0]['objects'], batch))
        
        img_ids = []
        for item in zip(batch, tips):
            img_ids.extend([item[0][1]] * item[1].size()[0])
        
        return {'imgs': torch.cat(imgs, dim=0), 'tips': torch.cat(tips, dim=0), 'objects': objects}, img_ids
    
    @staticmethod
    def raw_resolution_collate_fn(batch):
        imgs = list(map(lambda item: item[0]['imgs'].squeeze(0), batch))
        img_ids = []
        for item in zip(batch, imgs):
            img_ids.extend([item[0][1]] * item[1].size()[0])
        return {'imgs': imgs}, img_ids


def build_cc3m_dataset(
    tokenizer: Tokenizer,
    config_path=default_config_path,
    text_mode: str = 'once',
    raw_resolution: bool = False,
    return_objects: bool = True
):
    config = parse_yaml(config_path)
    image_processor = ValImageProcessor() if raw_resolution is False else RawResolutionImageProcessor()
    text_processor = CaptionProcessor(config['cc3m']['txt_mapper_path'], tokenizer, text_mode)
    return CC3MDataset(tokenizer, image_processor, text_processor, LocationEmbedding(), return_objects, config_path)


'''
------------------------------------------The VG Dataset.------------------------------------------
'''
class VGTextProcessor:
    
    def __init__(
        self,
        question_answer_path: str,
        tokenizer: Tokenizer
    ):
        self.tokenizer = tokenizer
        with open(question_answer_path) as f:
            self.txt_mapper = json.load(f)
        self.question_types = set(['what', 'how', 'where', 'who', 'why', 'when'])

    def process(self, img_id):
        # read qa
        qas = self.txt_mapper[str(img_id)]
        qa = random.choice(qas)
        question_type = self.get_question_type(qa['question'])
        target = self.tokenizer.concat_tokens([self.clean(qa['question']), self.clean(qa['answer'])], self.tokenizer.question_answer_sep_token)
        target = self.tokenizer.get_padded_tokens(target).unsqueeze(0)
        return target, question_type, qa['qa_id']

    def get_question_type(self, question):
        first_word = self.clean(self.tokenizer.tokenizer.tokenize(question)[0])
        return first_word if first_word in self.question_types else 'other'

    def clean(self, text: str):
        return text.replace('.', '').lower()


class QuestionRegionMapper:
    
    def __init__(
        self,
        mapper_path: str,
        general_caption_path: str,
        region_path: str
    ):
        with open(mapper_path) as f:
            self.qr_mapper = json.load(f)
        
        with open(region_path) as f:
            self.region_mapper = json.load(f)
        
        with open(general_caption_path) as f:
            self.general_caption = json.load(f)
    
    def __call__(self, img_id, qa_id):
        qa_id = str(qa_id)
        # map region to question(s)
        if qa_id in self.qr_mapper and str(self.qr_mapper[qa_id]) in self.region_mapper:
            region = self.region_mapper[str(self.qr_mapper[qa_id])]
            # caption, (x1, y1, x2, y2)
            return region['phrase'], (region['x'], region['y'], region['x'] + region['width'], region['y'] + region['height'])
        else:
            return self.general_caption[img_id], None


class CaptionTask:
    
    def __init__(
        self,
        image_region_mapper_path: str,
        general_caption_path: str
    ) -> None:
        with open(image_region_mapper_path) as f:
            self.image_region_mapper = json.load(f)

        with open(general_caption_path) as f:
            self.general_caption = json.load(f)

    def __call__(self, img_id):
        regions = self.image_region_mapper[str(img_id)]
        general = {
            'general': True,
            'phrase': self.general_caption[str(img_id)],
            'x': 0,
            'y': 0,
            'width': 0,
            'height': 0
        }
        regions.append(general)
        region = random.choice(regions)
        if region['x'] < 0 or region['y'] < 0 or region['width'] < 0 or region['height'] < 0:
            region = general
        return region


class VGDataset(Dataset):
    
    def __init__(
        self,
        text_processor: VGTextProcessor,
        image_processor: ImageProcessor,
        location_embedding: LocationEmbedding,
        question_region_mapper: QuestionRegionMapper,
        tokenizer: Tokenizer,
        caption_task: CaptionTask,
        image_path: str,
        id_path: str,
        config_path=default_config_path,
        filter_noise: bool = True
    ):
        super().__init__()
        config = parse_yaml(config_path)
        self.resolution = config['video_resolution']
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.location_embedding = location_embedding
        self.question_region_mapper = question_region_mapper
        self.caption_task = caption_task
        self.image_path = image_path
        self.tokenizer = tokenizer
        # read image id
        with open(id_path) as f:
            self.ids = json.load(f)
        # filter unmatched qa to region mapping
        if filter_noise is True:
            self.filter_noise()
    
    def __getitem__(self, index):
        img_id = self.ids[index]
        # read image
        img_path = os.path.join(self.image_path, '{}.jpg'.format(img_id))
        img = Image.open(img_path)
        # get image size: W, H
        img_size = img.size[0], img.size[1]
        img = self.image_processor.process(img).unsqueeze(0)
        # read text data
        target, question_type, qa_id = self.text_processor.process(img_id)
        # get region data
        caption, region = self.question_region_mapper(img_id, qa_id)
        # embed location
        location = self.location_embedding(self.resolution, img_size, region)
        # get tip tokens
        tip = self.tokenizer.concat_tokens([question_type, location, caption], self.tokenizer.task_prompt_sep_token)
        tip = self.tokenizer.get_padded_tokens(tip).unsqueeze(0)
        
        # get caption tip and target
        caption_task_region = self.caption_task(img_id)
        caption_task_region_caption = caption_task_region['phrase']
        caption_task_region_area = (
            caption_task_region['x'],
            caption_task_region['y'],
            caption_task_region['x'] + caption_task_region['width'],
            caption_task_region['y'] + caption_task_region['height']
        ) if 'general' not in caption_task_region else None
        caption_task_location = self.location_embedding(self.resolution, img_size, caption_task_region_area)
        caption_tip = self.tokenizer.concat_tokens([caption_task_location], self.tokenizer.task_prompt_sep_token)
        caption_tip = self.tokenizer.get_padded_tokens(caption_tip).unsqueeze(0)
        caption_target = self.tokenizer.tokenize(caption_task_region_caption).unsqueeze(0)
        return {
            'imgs': img,
            'tips': tip,
            'targets': target,
            'caption_tips': caption_tip,
            'caption_targets': caption_target,
            'region': self.generate_indicator(self.location_embedding.transfer_location(self.resolution, img_size, region), self.resolution),
            'caption_region': self.generate_indicator(self.location_embedding.transfer_location(self.resolution, img_size, caption_task_region_area), self.resolution)
        }, target, [str(img_id)] * tip.size()[0]
    
    @staticmethod
    def generate_indicator(region, resolution):
        indicator = torch.zeros(resolution, resolution, dtype=int)
        indicator[region[1]:region[3] + 1, region[0]:region[2] + 1] = 1
        return indicator
    
    def filter_noise(self):
        ids_to_remove = []
        for img_id in self.ids:
            qas = self.text_processor.txt_mapper[str(img_id)]
            qas_to_remove = []
            # match qa to region
            for qa in qas:
                qa_id = qa['qa_id']
                _, result = self.question_region_mapper(img_id, qa_id)
                if result is None:
                    qas_to_remove.append(qa)
            # remove whole img
            if len(qas) == len(qas_to_remove):
                ids_to_remove.append(img_id)
            # remove unmatched qas
            for _qa in qas_to_remove:
                qas.remove(_qa)
        for _id in ids_to_remove:
            self.ids.remove(_id)
        # count filtered data
        print('images left: {}'.format(len(self.ids)))
        total_qas = 0
        for img_id in self.ids:
            qas = self.text_processor.txt_mapper[str(img_id)]
            total_qas += len(qas)
        print('questions left: {}'.format(total_qas))

    def __len__(self):
        return len(self.ids)


def build_vg_dataset(
    mode: str,
    tokenizer: Tokenizer,
    config_path=default_config_path,
    filter_noise: bool = True
):
    img_processor_dict = {
        'train': TrainImageProcessor,
        'val': ValImageProcessor
    }
    config = parse_yaml(default_config_path)
    # mode: train / val
    items = config[mode]
    return VGDataset(
        VGTextProcessor(items['question_answer_path'], tokenizer),
        img_processor_dict[mode](),
        LocationEmbedding(),
        QuestionRegionMapper(items['mapper_path'], items['general_caption_path'], items['region_path']),
        tokenizer,
        CaptionTask(items['image_region_mapper_path'], items['general_caption_path']),
        items['img_path'],
        items['id_path'],
        config_path,
        filter_noise
    )


'''
------------------------------------------Some utils for building a dataset.------------------------------------------
'''
class CustomDistributedSampler(DistributedSampler):

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if self.drop_last:
            indices = indices[:self.total_size]
        indices = indices[self.rank:len(indices):self.num_replicas]
        return iter(indices)


def build_dataloader(dataset: Dataset, batch_size, shuffle: bool = True, collate_fn=None):
    config = parse_yaml(default_config_path)
    batch_size = batch_size // dist.get_world_size()
    sampler = CustomDistributedSampler(dataset, shuffle=shuffle)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate_fn, num_workers=config['num_workers'], pin_memory=config['pin_memory']), sampler


def reshape_tensor(batch):
    items = [
        {'item': 'imgs', 'length': 5},
        {'item': 'tips', 'length': 2},
        {'item': 'targets', 'length': 2},
        {'item': 'caption_targets', 'length': 2},
        {'item': 'caption_tips', 'length': 2}
    ]
    
    for item in items:
        if item['item'] in batch and len(batch[item['item']].size()) > item['length']:
            batch[item['item']] = torch.flatten(batch[item['item']], start_dim=0, end_dim=1)
    return batch
