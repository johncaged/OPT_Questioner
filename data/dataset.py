from model.bert_tokenizer import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from utils import default_config_path, parse_yaml
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


class ImageProcessor:
    
    def __init__(self):
        super().__init__()
        self.mean = [0.48145466, 0.4578275, 0.40821073] 
        self.std  = [0.26862954, 0.26130258, 0.27577711]
        self.resolution = 480
        self.transforms = NOTHING
    
    def process(self, img):
        img = img.convert('RGB')
        img = ToTensor()(img)
        return self.transforms(img).unsqueeze(0)


class TrainImageProcessor(ImageProcessor):
    
    def __init__(self):
        super().__init__()
        self.transforms = Compose([RandomResizedCrop(self.resolution, [0.8,1.0],[1.0,1.0]),
                                   RandomHorizontalFlip(),
                                   Normalize(self.mean,self.std)])


class ValImageProcessor(ImageProcessor):
    
    def __init__(self):
        super().__init__()
        self.transforms = Compose([Resize(self.resolution),
                                   CenterCrop(self.resolution),
                                   Normalize(self.mean,self.std)])


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
        self.ids = list(set(self.vqa.getImgIds()))
        self.data_size = len(self.ids)
        self.confident = confident
        self.tokenizer = tokenizer
        
        assert mode in ['once', 'all']
        # WARNING: If the mode is set to 'all', the batch size may vary during the iteration process.
        self.mode = mode
    
    def process(self, image_id, image_name):
        pass

    def get_image_id_through_index(self, index):
        return self.ids[index]
    
    def get_tokenized_text_pair(self, text1, text2):
        """Get a single tokenized text pair.
        """
        return self.tokenizer.tokenize(text1).unsqueeze(0), self.tokenizer.tokenize(text2).unsqueeze(0)


class QuestionAnswerProcessor(TextProcessor):
    """Get question-answer pair for question generation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def process(self, image_id, image_name):
        # get questions
        question_ids = self.vqa.getQuesIds(imgIds=image_id)
        
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
            question, answer = self.get_single_pair(self.vqa.qqa[q_id]['question'], temp['answers'])
            questions.append(question)
            answers.append(answer)
        return questions, answers
    
    def get_single_pair(self, question, answers):
        if self.confident:
            temp = filter(lambda answer: answer.answer_confidence == 'yes', answers)
            if len(temp) >= 1:
                answers = temp
        answer = random.choice(answers)
        return self.get_tokenized_text_pair(question, answer['answer'])


class QuestionCaptionProcessor(TextProcessor):
    """Get question-caption pair for question generation.
    """
    
    def __init__(self, caption_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(caption_path) as f:
            self.captions = json.load(f)
    
    def process(self, image_id, image_name):
        # get captions
        img_captions = self.captions[image_name]
        # get questions
        question_ids = self.vqa.getQuesIds(imgIds=image_id)
        
        questions, captions = None, None
        
        # image -> n * questions & m captions
        if self.mode == 'once':
            # random select 1 question from n questions
            q_id = random.choice(question_ids)
            questions, captions = self.get_multiple_pairs([q_id], img_captions)
        elif self.mode == 'all':
            questions, captions = self.get_multiple_pairs(question_ids, img_captions)
        return questions, captions

    def get_multiple_pairs(self, q_ids, img_captions):
        """Get multiple question-caption pairs through an array of ids.
        """
        questions = []
        captions = []
        
        # choose one caption for every single question.
        for q_id in q_ids:
            caption = random.choice(img_captions)
            question, caption = self.get_tokenized_text_pair(self.vqa.qqa[q_id]['question'], caption)
            questions.append(question)
            captions.append(caption)
        return questions, captions


class VQADataset(Dataset):
    
    def __init__(
        self,
        image_processor: ImageProcessor,
        text_processor: TextProcessor,
        image_path: str,
        image_prefix: str,
        auto_regressive: bool = False
    ):
        super().__init__()
        self.image_path = image_path
        self.image_prefix = image_prefix
        self.image_processor = image_processor
        self.text_processor = text_processor
        self.auto_regressive = auto_regressive

    def __getitem__(self, index):
        # get image
        image_id = self.text_processor.get_image_id_through_index(index)
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
        imgs = img.repeat(targets.size()[0], *([1] * 4))
        # WARNING: the two targets variable are referring to the same memory address.
        return {'imgs': imgs, 'tips': tips, 'targets': targets, 'auto_regressive': self.auto_regressive}, targets, [str(image_id)] * targets.size()[0]

    def __len__(self):
        return self.text_processor.data_size


def build_dataset(dataset_type: str, mode: str, tokenizer: Tokenizer, auto_regressive: bool = False, text_mode: str = 'once'):
    img_processor_dict = {
        'train': TrainImageProcessor,
        'val': ValImageProcessor
    }
    config = parse_yaml(default_config_path)
    # mode: train / val
    items = config[mode]
    # text processor
    text_processor = QuestionAnswerProcessor(items['question'], items['annotation'], tokenizer, text_mode) if dataset_type == 'answer' else \
        QuestionCaptionProcessor(items['caption'], items['question'], items['annotation'], tokenizer, text_mode)
    return VQADataset(img_processor_dict[mode](), text_processor, items['image'], items['image_prefix'], auto_regressive)


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


def build_dataloader(dataset: Dataset, batch_size, shuffle: bool = True):
    config = parse_yaml(default_config_path)
    batch_size = batch_size // dist.get_world_size()
    sampler = CustomDistributedSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=config['num_workers'], pin_memory=config['pin_memory'])
