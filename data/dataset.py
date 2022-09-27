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
        self.text_encoder = CLIPText(text_encoder)
        self.k = 0.4
    
    def process(self, image_id, image_name):
        # get captions
        img_captions = self.captions[image_name]
        # get questions
        question_ids = self.vqa.getQuesIds(imgIds=image_id)
        
        questions, captions = None, None
        
        # image -> n * questions & m captions
        if self.mode == 'once':
            # random select 1 question from n questions
            img_caption = random.choice(img_captions)
            questions, captions = self.get_multiple_pairs(list(map(lambda q_id: self.vqa.qqa[q_id]['question'], question_ids)), [img_caption])
        elif self.mode == 'all':
            questions, captions = self.get_multiple_pairs(list(map(lambda q_id: self.vqa.qqa[q_id]['question'], question_ids)), img_captions)
        return questions, captions

    def get_multiple_pairs(self, img_questions, img_captions):
        """Get multiple question-caption pairs through an array of ids.
        """
        questions = []
        captions = []
        
        # choose one caption for every single question.
        for img_caption in img_captions:
            # choose through clip cosine similarity.
            question = self.choose(img_questions, img_caption)
            question, caption = self.get_tokenized_text_pair(question, img_caption)
            questions.append(question)
            captions.append(caption)
        return questions, captions
    
    def choose(self, questions, caption):
        caption_token = tokenize(caption)
        question_tokens = tokenize(questions)
        
        class TrainingMode:
            def __init__(self, model):
                self.model = model
                self.training = model.training
            
            def __enter__(self):
                self.model.eval()
            
            def __exit__(self, *args):
                self.model.train(self.training)
        
        with torch.no_grad():
            with TrainingMode(self.text_encoder):
                caption_feature, text = self.text_encoder.encode_text(caption_token, casual=False)
                caption_feature = caption_feature[torch.arange(caption_feature.shape[0]), torch.argmax(text, dim=-1)]
                caption_feature = caption_feature / caption_feature.norm(dim=1, keepdim=True)
                
                question_features, text = self.text_encoder.encode_text(question_tokens, casual=False)
                question_features = question_features[torch.arange(question_features.shape[0]), torch.argmax(text, dim=-1)]
                question_features = question_features / question_features.norm(dim=1, keepdim=True)
                
                similarity = torch.mm(caption_feature, question_features.permute(1, 0))
                sorted_similarity, indices = torch.sort(similarity)
                sim_rank = indices.tolist()[0]
                del similarity, sorted_similarity, question_features, caption_feature, text, caption_token, question_tokens
        length = max(int(self.k * len(sim_rank)), 2)
        question_index = random.choice(sim_rank[0:length])
        return questions[question_index]


class VQADataset(Dataset):
    
    def __init__(
        self,
        image_processor: ImageProcessor,
        text_processor: TextProcessor,
        image_path: str,
        image_prefix: str
    ):
        super().__init__()
        self.image_path = image_path
        self.image_prefix = image_prefix
        self.image_processor = image_processor
        self.text_processor = text_processor

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
        return {'imgs': imgs, 'tips': tips, 'targets': targets}, targets, [str(image_id)] * targets.size()[0]

    def __len__(self):
        return self.text_processor.data_size


def build_dataset(
    dataset_type: str,
    mode: str,
    tokenizer: Tokenizer,
    text_mode: str = 'once',
    confident: bool = False,
    multiple_choice_answer: bool = True,
    quesTypes: list = [],
    ansTypes: list = [],
    text_encoder = None,
    k: int = 2
):
    img_processor_dict = {
        'train': TrainImageProcessor,
        'val': ValImageProcessor
    }
    config = parse_yaml(default_config_path)
    # mode: train / val
    items = config[mode]
    # text processor
    text_processor = QuestionAnswerProcessor(multiple_choice_answer, quesTypes, ansTypes, items['question'], items['annotation'], tokenizer, text_mode, confident) if dataset_type == 'answer' else \
        QuestionCaptionProcessor(items['caption'], text_encoder, k, items['question'], items['annotation'], tokenizer, text_mode)
    return VQADataset(img_processor_dict[mode](), text_processor, items['image'], items['image_prefix'])


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
