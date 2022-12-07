import torch.nn as nn
import torch
from .clip import build_model
from .transformer import GELU
from .bert import BertModel, BertConfig
from apex.normalization.fused_layer_norm import FusedLayerNorm
import os
from utils import default_config_path, parse_yaml
from data.dataset import Tokenizer, reshape_tensor
import copy
import numpy as np
import random
import torch.nn.functional as F
from utils import ToCuda
from torch_lib.util import NOTHING


class QuestionerModule(nn.Module):
    """
    Questioner Module used to setup model structure and load pretrained weights.
    """
    
    def __init__(self, config_path=default_config_path):
        super().__init__()
        self.video_dim = 1024
        self.multimodal_dim = 768
        self.build_model_structure(config_path=config_path)
    
    def build_model_structure(self, config_path=default_config_path):
        """Build model structure using clip initial weight and bert initial weight.

        Args:
            config_path (str, optional): config path. Defaults to default_config_path.
        """
        config = parse_yaml(config_path)
        
        # Init clip structure using initial weight.
        clip_initial_config = torch.jit.load(os.path.join(config['initial_path'], config['clip_initial_path']), map_location='cpu').state_dict()
        self.clip = build_model(clip_initial_config, config['video_resolution'], config['gradient_checkpointing']).float()
        
        # Init bert structure using initial config.
        bert_config = BertConfig.from_json_file(os.path.join(config['initial_path'], config['bert_config_path']))
        
        visual_config = copy.deepcopy(bert_config)
        
        bert_config.checkpointing = config['gradient_checkpointing']
        visual_config.checkpointing = config['gradient_checkpointing']
        
        # Inherit from the best configuration of OPT-Three.
        visual_config.max_visual_frame_embeddings = 10
        visual_config.has_decoder = False

        # vqgan-f16 for initialization
        visual_config.visual_vocab_size = 16384 + 4  # (8192 + [BOI] + [BOV] + [MASKV] + MASK[I])
        visual_config.tokens_per_frame = 64
        visual_config.image_token_max_position_embeddings = 256
        visual_config.video_token_max_position_embeddings = 640  # 64 * 10
        
        # cross attn: Inherit from the best configuration.
        bert_config.has_cross_attn = True
        bert_config.key_value_dim = self.multimodal_dim
        
        self.bert = BertModel(bert_config, visual_config)
        # The variable 'self.bert.embeddings.word_embeddings.weight' is only used to get the shape of the weight matrix.
        embedding_weight = self.bert.embeddings.word_embeddings.weight
        self.cls_head = PredictionHead(embedding_weight.size(1), embedding_weight.size(0))
        
        # video dim multimodal adapter
        self.video_feature_adapter = nn.Sequential(
            nn.Linear(self.video_dim, self.multimodal_dim),
            FusedLayerNorm(self.multimodal_dim, eps=1e-12)
        )
        # video frame embedding
        self.video_frame_embedding = nn.Parameter(0.02 * torch.randn(1, 32, self.multimodal_dim))
        # video type embedding
        self.video_type_embedding = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim))
        
        # whether to use soft prompt
        self.soft_prompt = config['soft_prompt']

    def load_pretrained_weights(self, config_path=default_config_path):
        """Load pretrained bert and clip respectively using config file.

        Args:
            config_path (str, optional): config path. Defaults to default_config_path.
        """
        config = parse_yaml(config_path)
        pretrained_path = config['pretrained_path']
        # load clip and bert weights
        clip_weight = torch.load(os.path.join(pretrained_path, config['clip_pretrained_path']), map_location='cpu')
        if config['pretrained_resolution'] != config['video_resolution']:
            self.adapt_clip_resolution(clip_weight, config['video_resolution'])
        self._load_state_dict(self.clip, 'clip', clip_weight)
        self._load_state_dict(self.bert, 'bert', torch.load(os.path.join(pretrained_path, config['bert_pretrained_path']), map_location='cpu'))
        self._load_state_dict(self.cls_head, 'cls_head', torch.load(os.path.join(pretrained_path, config['cls_head_pretrained_path']), map_location='cpu'))
        self._load_state_dict(self.video_feature_adapter, 'video_feature_adapter', torch.load(os.path.join(pretrained_path, config['video_feature_adapter_pretrained_path']), map_location='cpu'))
        self.video_frame_embedding = nn.Parameter(torch.load(os.path.join(pretrained_path, config['video_frame_embedding_pretrained_path']), map_location='cpu'))
        self.video_type_embedding = nn.Parameter(torch.load(os.path.join(pretrained_path, config['video_type_embedding_pretrained_path']), map_location='cpu'))

    @staticmethod
    def _load_state_dict(module, module_name, *args, **kwargs):
        missing_keys, unexpected_keys = module.load_state_dict(*args, **kwargs, strict=False)
        print('{}:'.format(module_name))
        print('missing keys: {}'.format(missing_keys))
        print('unexpected keys: {}'.format(unexpected_keys))

    def adapt_clip_resolution(self, weight, video_resolution):
        vision_width = weight["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in weight.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = weight["visual.conv1.weight"].shape[-1]
        grid_size = round((weight["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        
        src = weight["visual.positional_embedding"]
        src_cls = src[0:1]
        src_oth = src[1:]
        new_grid_size = video_resolution // vision_patch_size
        src_oth = F.interpolate(src_oth.reshape(grid_size, grid_size, vision_width).permute(2, 0, 1).unsqueeze(0), (new_grid_size, new_grid_size), mode='bilinear')
        src_oth = src_oth[0].permute(1, 2, 0).reshape(-1, 1024)
        tgt = torch.cat((src_cls, src_oth), dim=0)
        weight["visual.positional_embedding"] = tgt

    def load_cls_from_bert_initial(self, bert_weight):
        # WARNING: load from bert initial weight, not from pretrained weight.
        bert_weight = {k.replace('bert.', '').replace('gamma', 'weight').replace('beta', 'bias') : v for k, v in bert_weight.items()}
        self.cls_head.load_weight(torch.nn.Parameter(bert_weight['embeddings.word_embeddings.weight']))
        cls_head_weight = {
            'dense.weight': bert_weight['cls.predictions.transform.dense.weight'],
            'dense.bias': bert_weight['cls.predictions.transform.dense.bias'],
            'layernorm.weight': bert_weight['cls.predictions.transform.LayerNorm.weight'],
            'layernorm.bias': bert_weight['cls.predictions.transform.LayerNorm.bias'],
            'decoder.weight': bert_weight['cls.predictions.decoder.weight'],
            'decoder.bias': bert_weight['cls.predictions.bias']
        }
        self._load_state_dict(self.cls_head, 'cls_head', cls_head_weight)
        del bert_weight, cls_head_weight


class BaseQuestioner(QuestionerModule):
    """
    Base Questioner used to define general operations such as decoding output logits to text and embedding video inputs, etc.
    """
    
    def __init__(
        self,
        tokenizer: Tokenizer,
        questioner_adapter,
        auto_regressive: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.masker = TokenMasker(mask_token=tokenizer.mask_token, range_start=106, range_end=30522, question_answer_sep_token=tokenizer.question_answer_sep_token)
        self.tokenizer = tokenizer
        self.questioner_adapter = questioner_adapter
        self.auto_regressive = auto_regressive
        self._forward = None
    
    def get_task_prompt(self, content, batch_size):
        torch_convert = lambda item: torch.tensor(item).unsqueeze(0).expand(batch_size, -1).long()
        
        prompt = self.tokenizer.tokenize_without_padding(content)
        return ToCuda(torch.cat([torch_convert(prompt), torch_convert([self.tokenizer.task_prompt_sep_token])], dim=1))
    
    def forward_video(self, video, region=None):
        b, n, _, h, w = video.shape
        video_output = self.clip.encode_image(video.reshape(b * n, 3, h, w), region)
        video_output = video_output.reshape(b, -1, *video_output.shape[-2:])
        return self.get_video_multimodal_embedding(video_output)

    def get_video_multimodal_embedding(self, video):
        b, n, x, c = video.shape

        if hasattr(self, 'video_feature_adapter'):
            video = self.video_feature_adapter(video)
        video = video + self.video_frame_embedding[:, :video.shape[1], :].unsqueeze(-2)
        video = video.reshape(b, -1, self.multimodal_dim)
        video = video + self.video_type_embedding
        return video

    def forward(self, batch):
        batch: dict = reshape_tensor(batch)
        img, tip, target = ToCuda(batch['imgs']), ToCuda(batch['tips' if self._forward != 'caption' else 'caption_tips']) , ToCuda(batch['targets' if self._forward != 'caption' else 'caption_targets'])
        region = batch.setdefault('region', None) if self._forward != 'caption' else batch.setdefault('caption_region', None)

        if self.auto_regressive is False:
            # random mask the target.
            target, labels = self.masker(target, 0.6, answer_mask=self._forward == 'answer')
            if self._forward == 'answer':
                tip[:, 1] = self.tokenizer.answer_type_mask_token
            video_input = self.forward_video(img, region)
            output_txt = self.video_language_process(video_input, tip, target)
            output_txt = output_txt[labels != -1]
            prediction_scores = self.cls_head(output_txt)
            del output_txt, img, tip, target
            return prediction_scores, labels[labels != -1]
            
        else:
            # using cache to optimize performance
            if 'video_input' not in batch:
                batch['video_input'] = self.forward_video(img, region)
            video_input = batch['video_input']
            output_txt = self.video_language_process(video_input, tip, target)
            output_txt = output_txt[:, -1]
            prediction_scores = self.cls_head(output_txt)
            del output_txt, img, tip, target
            return prediction_scores, None

    def video_language_process(self, video_input, tip, target):
        # get task prompt
        batch_size = tip.shape[0]
        task_prompt = self.get_task_prompt(self.questioner_adapter.get_task(self._forward), batch_size)
        task_prompt = torch.cat((tip[:, 0:1], task_prompt, tip[:, 1:]), dim=1)
        # forward multimodal
        original_output = self.bert(target, task_prompt, video_input, None, casual=True)
        output_txt = original_output[:, :target.shape[1], :]
        return output_txt

    def get_logits(self, batch, state):
        batch_size = self.get_batch_size(batch)
        
        masked_tokens = ToCuda(torch.zeros(batch_size, 1, dtype=torch.long)).fill_(self.tokenizer.mask_token)
        bos_token = ToCuda(torch.zeros(batch_size, 1, dtype=torch.long)).fill_(self.tokenizer.cls_token)
        txt_tokens = torch.cat((state, masked_tokens), dim=1) if state is not None else masked_tokens
        txt_tokens = torch.cat((bos_token, txt_tokens), dim=1)
        
        batch['targets' if self._forward != 'caption' else 'caption_targets'] = txt_tokens
        return self(batch)[0]
    
    def get_batch_size(self, batch):
        return batch['imgs'].shape[0]


class QuestionerWithAnswer:
    
    def __init__(self):
        super().__init__()
    
    def get_task(self):
        return 'question generation with visual and answer cues'


class QuestionerWithCaption:

    def __init__(self):
        super().__init__()
    
    def get_task(self):
        return 'question generation with visual and caption cues'


class MultiStageAdapter:
    
    def get_task(self, _forward):
        if _forward == 'caption':
            return 'generate region caption'
        else:
            return 'generate question answer pair'


class PredictionHead(nn.Module):
    """Prediction output head that transforms feature vector to vocabulary scores.
    """
    
    def __init__(self, hidden_size, vocab_size, embedding_weights=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = GELU()
        self.layernorm = FusedLayerNorm(self.hidden_size, eps=1e-12)
        self.decoder = nn.Linear(self.hidden_size, self.vocab_size)
        self.load_weight(embedding_weights)

    def forward(self, sequence_output):
        sequence_output = self.dense(sequence_output)
        sequence_output = self.activation(sequence_output)
        sequence_output = self.layernorm(sequence_output)
        prediction_scores = self.decoder(sequence_output) 
        return prediction_scores

    def load_weight(self, weight):
        if weight is not None:
            self.decoder.weight = weight


class TokenMasker:

    def __init__(self, mask_token = -1, range_start=-1, range_end=-1, question_answer_sep_token=-1):
        super().__init__()
        self.mask_token = mask_token
        self.range = [range_start, range_end]
        self.question_answer_sep_token = question_answer_sep_token

    def __call__(self, tokens, mask_prob, answer_mask: bool = False):
        tokens, labels = self.perform_mask(tokens, mask_prob, answer_mask=answer_mask)
        return tokens, labels

    def perform_mask(self, tokens, mask_prob, answer_mask: bool = False):
        tokens = np.array(tokens.clone().detach().cpu().numpy())
        
        # generate indicator first:
        mask_indicator = np.zeros(tokens.shape, dtype=np.int64)
        for i in range(len(mask_indicator)):
            while all(mask_indicator[i] == 0):
                answer_part = False
                for j in range(1, len(mask_indicator[0])):
                    if tokens[i][j - 1] == self.question_answer_sep_token:
                        answer_part = True
                    
                    # mask all answer tokens.
                    _mask_answer_condition = answer_mask is True and answer_part is True
                    # random mask question tokens.
                    _mask_question_condition = answer_mask is False and answer_part is False and random.random() < mask_prob
                    if tokens[i][j] != 0 and (_mask_answer_condition or _mask_question_condition):
                        mask_indicator[i][j] = 1

        labels = -np.ones(tokens.shape, dtype=np.int64)
        for i in range(tokens.shape[0]):
            for j in range(tokens.shape[1]):
                if mask_indicator[i][j] == 1 :
                    src_token = tokens[i][j]
                    prob = random.random()   # e-6 too much time
                    if prob < 0.8:
                        tokens[i][j] = self.mask_token  # e-6 have no idea why too much
                    elif prob < 0.9: 
                        tokens[i][j] = random.choice(list(range(*self.range)))
                    # tokens[i][j] = self.mask_token
                    labels[i][j] = src_token

        tokens = ToCuda(torch.from_numpy(tokens).long())
        labels = ToCuda(torch.from_numpy(labels).long())

        return tokens, labels


class TextSampler(nn.Module):
    
    def __init__(self, module: BaseQuestioner, end_token=None):
        super().__init__()
        self.module: BaseQuestioner = module
        self.end_token = end_token if end_token is not None else self.module.tokenizer.sep_token
    
    def sample(self, batch):
        pass
    
    def forward(self, batch):
        return self.sample(batch)

    def clone_batch(self, batch):
        item = {}
        for key, value in batch.items():
            item[key] = value.clone() if torch.is_tensor(value) else value
        return item


class BeamSampler(TextSampler):
    
    def __init__(self, module, beam_size=3, end_token=None):
        super().__init__(module, end_token)
        self.beam_size = beam_size
    
    def sample(self, batch):
        batch = self.clone_batch(batch)
        batch = reshape_tensor(batch)
        
        max_generation_len = 30

        beam_size = self.beam_size
        batch_size = self.module.get_batch_size(batch)
        seq_logprob = ToCuda(torch.zeros((batch_size, 1, 1)))
        log_probs = []
        selected_words = None
        seq_mask = ToCuda(torch.ones((batch_size, beam_size, 1)))

        state = None
        cache = {'key': {}, 'value': {}, 'attn_masks': None}
        outputs = []
        for t in range(max_generation_len):
            cur_beam_size = 1 if t == 0 else beam_size

            word_logits = self.module.get_logits(batch, state)
            word_logprob = F.log_softmax(word_logits, dim=1)

            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            candidate_logprob = seq_logprob + word_logprob

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != self.end_token).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            selected_idx, selected_logprob = self.select(batch_size, beam_size, t, candidate_logprob)
            selected_beam = selected_idx // candidate_logprob.shape[-1]
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1, 1)
            
            if state is not None:
                state = self._adjust_tensor(batch_size, beam_size, state, selected_beam)
                state = torch.cat((state,selected_words),dim = 1)
                for k in cache['key']:
                    cache['key'][k] = self._adjust_tensor(batch_size, beam_size, cache['key'][k], selected_beam)
                for k in cache['value']:
                    cache['value'][k] = self._adjust_tensor(batch_size, beam_size, cache['value'][k], selected_beam)
                cache['attn_masks'] = self._adjust_tensor(batch_size, beam_size, cache['attn_masks'], selected_beam)
            else:
                state = selected_words
                for k in cache['key']:
                    cache['key'][k] = self.expand_tensor(cache['key'][k], beam_size)
                for k in cache['value']:
                    cache['value'][k] = self.expand_tensor(cache['value'][k], beam_size)
                cache['attn_masks'] = self.expand_tensor(cache['attn_masks'], beam_size)

            if t == 0:
                if batch['imgs'] is not None:
                    batch['imgs'] = self.expand_tensor(batch['imgs'], beam_size)
                if batch['tips'] is not None:
                    batch['tips'] = self.expand_tensor(batch['tips'], beam_size)
 
        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, max_generation_len))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, max_generation_len))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]

        return outputs

    def select(self, batch_size, beam_size, t, candidate_logprob):
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(batch_size, -1), -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :beam_size], selected_idx[:, :beam_size]
        return selected_idx, selected_logprob

    def _adjust_tensor(self, batch_size, beam_size, tensor, selected_beam):
        if tensor is None:
            return tensor
        if tensor.dim() == 4:
            b,h,n,c= tensor.shape
            tensor = torch.gather(tensor.view(batch_size, beam_size,h, n,c), 1, \
            selected_beam.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(batch_size, beam_size,h,n,c))
            tensor = tensor.reshape(b,h,n,c)
        if tensor.dim() == 3:
            b,n,c= tensor.shape
            tensor = torch.gather(tensor.view(batch_size, beam_size, n,c), 1, \
            selected_beam.unsqueeze(-1).unsqueeze(-1).expand(batch_size, beam_size,n,c))
            tensor = tensor.reshape(b,n,c)
        elif tensor.dim() == 2:
            b,n = tensor.shape
            tensor = torch.gather(tensor.view(batch_size, beam_size, n), 1, \
            selected_beam.unsqueeze(-1).expand(batch_size, beam_size, n))
            tensor = tensor.reshape(b,n)
        return tensor
    
    def expand_tensor(self, tensor, size, dim=1):
        if size == 1 or tensor is None:
            return tensor
        tensor = tensor.unsqueeze(dim)
        tensor = tensor.expand(list(tensor.shape[:dim]) + [size] + list(tensor.shape[dim + 1:])).contiguous()
        tensor = tensor.view(list(tensor.shape[:dim - 1]) + [-1] + list(tensor.shape[dim + 1:]))
        return tensor


class TopKSampler(TextSampler):
    
    def __init__(self, module, k=10, end_token=None):
        super().__init__(module, end_token)
        self.k = k
    
    def sample(self, batch):
        batch = self.clone_batch(batch)
        batch = reshape_tensor(batch)
        
        max_generation_len = 30
        batch_size = self.module.get_batch_size(batch)
        sents = ToCuda(torch.zeros((batch_size, max_generation_len), dtype=torch.long).fill_(self.end_token))

        unfinished = ToCuda(torch.ones(batch_size, dtype=torch.bool))

        state = None
        for t in range(max_generation_len):
            logits = self.module.get_logits(batch, state)

            logits = logits.scatter(1, (-logits).topk(logits.shape[1] - self.k, dim=1)[1], -10000)
            probs_t = F.softmax(logits, dim=1)
            wt = torch.multinomial(probs_t, 1)

            wt = wt.view(-1).long()
            unfinished = unfinished * (wt != self.end_token)
            wt = wt * unfinished.type_as(wt) + (1 - unfinished.type_as(wt)) * self.end_token
            sents[:, t] = wt

            state = wt.unsqueeze(1) if state is None else torch.cat((state, wt.unsqueeze(1)), dim=1)

            if unfinished.sum() == 0:
                break
        
        return sents


class TwoStageSampler(TextSampler):
    
    def __init__(self, module, base_k=5, end_token=None, question_answer_sep_token=None, answer_type_mask_token=None):
        super().__init__(module, end_token)
        self.base_k = base_k
        self.question_answer_sep_token = question_answer_sep_token
        self.answer_type_mask_token = answer_type_mask_token
    
    def sample(self, batch):
        batch = self.clone_batch(batch)
        batch = reshape_tensor(batch)
        
        max_generation_len = 30
        batch_size = self.module.get_batch_size(batch)
        # output sentence placeholder and unfinished flag.
        sentences = ToCuda(torch.zeros((batch_size, max_generation_len), dtype=torch.long).fill_(self.end_token))
        unfinished = ToCuda(torch.ones(batch_size, dtype=torch.bool))
        # question part flag.
        question_part = ToCuda(torch.ones(batch_size, dtype=torch.bool))
        # answer probability sum total and answer token length count(to compute average probability).
        prob_sum = ToCuda(torch.zeros(batch_size))
        len_count = ToCuda(torch.zeros(batch_size))
        # answer type mask token
        answer_type_mask_token = torch.ones(batch_size).type_as(batch['tips']) * self.answer_type_mask_token
        # current output state.
        state = None
        
        for t in range(max_generation_len):
            # mask answer type if answer part.
            batch['tips'][:, 1] = answer_type_mask_token * (1 - question_part.type_as(batch['tips'])) + batch['tips'][:, 1] * question_part.type_as(batch['tips'])
            logits = self.module.get_logits(batch, state)

            prob_sum, len_count = prob_sum.type_as(logits), len_count.type_as(logits)

            # greedy sample
            greedy_wt = torch.argmax(logits, dim=1).view(-1).long()
            # question part flag and unfinished flag
            question_part = question_part * (greedy_wt != self.question_answer_sep_token)
            unfinished = (unfinished * (greedy_wt != self.end_token)) | question_part

            # topk sample
            # clone logits before changing its values.
            logits_original = logits.clone()
            # topk should not sample [SEP] token or [unused1] token.
            eps = float(torch.min(logits).clone().detach().cpu()) - 1
            logits[:, self.end_token] = eps
            logits[:, self.question_answer_sep_token] = eps
            topk_index = logits.topk(self.base_k, dim=1)[1]
            topk_logits = logits.gather(1, topk_index)
            topk_probs = F.softmax(topk_logits, dim=1)
            topk_wt = torch.gather(topk_index, 1, torch.multinomial(topk_probs, 1)).view(-1).long()
            
            # select token
            wt = (topk_wt * question_part.type_as(topk_wt) + (1 - question_part.type_as(topk_wt)) * greedy_wt) * unfinished.type_as(topk_wt) + (1 - unfinished.type_as(topk_wt)) * self.end_token
            sentences[:, t] = wt

            state = wt.unsqueeze(1) if state is None else torch.cat((state, wt.unsqueeze(1)), dim=1)

            # compute average probability of predicted answers.
            answer_part = unfinished.type_as(logits_original) * (1 - question_part.type_as(logits_original))
            len_count += answer_part
            prob_sum += answer_part * torch.gather(F.softmax(logits_original, dim=1), 1, greedy_wt.unsqueeze(1)).squeeze(1)

            if unfinished.sum() == 0:
                break
        
        avg_prob = prob_sum / len_count
        avg_prob = torch.where(torch.isnan(avg_prob), torch.zeros(1).type_as(avg_prob), avg_prob)
        
        return sentences, avg_prob


class ModelWrapper(nn.Module):
    
    def __init__(self, module):
        super().__init__()
        self.module = module
