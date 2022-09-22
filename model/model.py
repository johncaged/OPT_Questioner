import torch.nn as nn
import torch
from .clip import build_model
from .transformer import GELU
from .bert import BertModel, BertConfig
from apex.normalization.fused_layer_norm import FusedLayerNorm
import os
from utils import default_config_path, parse_yaml
from data.dataset import Tokenizer
import copy
import numpy as np
import random
import torch.nn.functional as F
from utils import ToCuda


class Questioner(nn.Module):

    def __init__(self, tokenizer: Tokenizer, config_path=default_config_path):
        super().__init__()
        self.video_dim = 1024
        self.multimodal_dim = 768
        self.build_model_structure(config_path=config_path)
        self.masker = TokenMasker(mask_token=tokenizer.mask_token, range_start=106, range_end=30522)
        self.tokenizer = tokenizer

    def forward(self, batch):
        batch = self.reshape_tensor(batch)
        img, tip, target, auto_regressive = ToCuda(batch['imgs']), ToCuda(batch['tips']), ToCuda(batch['targets']), batch['auto_regressive']
        auto_regressive = torch.all(auto_regressive == False).item() is False

        if auto_regressive is False:
            # random mask to the target(also called 'question')
            target, labels = self.masker(target, 0.6)

        output_txt = self.video_language_process(img, tip, target)

        if auto_regressive is False:
            output_txt = output_txt[labels != -1]
        else:
            output_txt = output_txt[:, -1]

        prediction_scores = self.cls_head(output_txt)
        return prediction_scores, (labels[labels != -1] if auto_regressive is False else None)
    
    def video_language_process(self, img, tip, target):
        # get encoded and embedded img features
        video_input = self.forward_video(img)
        video_input = self.get_video_multimodal_embedding(video_input)
        # TODO: soft prompt
        # get task prompt
        batch_size = target.shape[0]
        task_prompt = self.get_task_prompt('question generation with visual and answer cues', batch_size)
        task_prompt = torch.cat((tip[:, 0:1], task_prompt, tip[:, 1:]), dim=1)
        # forward multimodal
        original_output = self.bert(target, task_prompt, video_input, None, casual=True)
        output_txt = original_output[:, :target.shape[1], :]
        return output_txt
    
    def reshape_tensor(self, batch):
        if 'imgs' in batch and len(batch['imgs'].size()) > 5:
            batch['imgs'] = torch.flatten(batch['imgs'], start_dim=0, end_dim=1)
        if 'tips' in batch and len(batch['tips'].size()) > 2:
            batch['tips'] = torch.flatten(batch['tips'], start_dim=0, end_dim=1)
        if 'targets' in batch and len(batch['targets'].size()) > 2:
            batch['targets'] = torch.flatten(batch['targets'], start_dim=0, end_dim=1)
        return batch
    
    def get_task_prompt(self, content, batch_size):
        prompt = self.tokenizer.tokenize_without_padding(content)
        # TODO: optimize
        return ToCuda(torch.tensor(prompt).unsqueeze(0).expand(batch_size, -1).long())
    
    def forward_video(self, video):
        b, n, _, h, w = video.shape
        video_output = self.clip.encode_image(video.reshape(b * n, 3, h, w))
        video_output = video_output.reshape(b, -1, *video_output.shape[-2:])
        return video_output

    def get_video_multimodal_embedding(self, video):
        b, n, x, c = video.shape
       
        if hasattr(self, 'video_feature_adapter'):
            video = self.video_feature_adapter(video)
        video = video + self.video_frame_embedding[:, :video.shape[1], :].unsqueeze(-2)
        video = video.reshape(b, -1, self.multimodal_dim)
        video = video + self.video_type_embedding
        return video

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
        self.clip.load_state_dict(clip_weight)
        self.bert.load_state_dict(torch.load(os.path.join(pretrained_path, config['bert_pretrained_path']), map_location='cpu'))
        self.cls_head.load_state_dict(torch.load(os.path.join(pretrained_path, config['cls_head_pretrained_path']), map_location='cpu'))
        self.video_feature_adapter.load_state_dict(torch.load(os.path.join(pretrained_path, config['video_feature_adapter_pretrained_path']), map_location='cpu'))
        self.video_frame_embedding = nn.Parameter(torch.load(os.path.join(pretrained_path, config['video_frame_embedding_pretrained_path']), map_location='cpu'))
        self.video_type_embedding = nn.Parameter(torch.load(os.path.join(pretrained_path, config['video_type_embedding_pretrained_path']), map_location='cpu'))

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
        self.cls_head.load_state_dict(cls_head_weight)
        del bert_weight, cls_head_weight


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


class TokenMasker(nn.Module):

    def __init__(self, mask_token = -1, range_start=-1, range_end=-1):
        super().__init__()
        self.mask_token = mask_token
        self.range = [range_start, range_end]

    def forward(self, tokens, mask_prob):
        tokens = tokens.clone()
        tokens, labels = self.perform_mask(tokens, mask_prob)
        return tokens, labels

    def perform_mask(self, tokens, mask_prob):
        tokens = np.array(tokens.cpu().numpy())

        # generate indicator first:
        mask_indicator = np.zeros(tokens.shape, dtype=np.int64)
        for i in range(len(mask_indicator)):
            while all(mask_indicator[i] == 0):
                for j in range(1, len(mask_indicator[0])):
                    if tokens[i][j]!=0 and random.random() < mask_prob:
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


class QuestionerGenerator(Questioner):
    
    def __init__(self, beam_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beam_size = beam_size

    def get_logits(self, batch, state):
        batch_size = self.get_batch_size(batch)
        
        masked_tokens = ToCuda(torch.zeros(batch_size, 1, dtype=torch.long)).fill_(self.tokenizer.mask_token)
        bos_token = ToCuda(torch.zeros(batch_size, 1, dtype=torch.long)).fill_(self.tokenizer.cls_token)
        txt_tokens = torch.cat((state, masked_tokens), dim=1) if state is not None else masked_tokens
        txt_tokens = torch.cat((bos_token, txt_tokens), dim=1)
        
        batch['targets'] = txt_tokens
        return super().__call__(batch)[0]

    def __call__(self, *args, **kwargs):
        return self.decode_beam(*args, **kwargs)

    def decode_beam(self, batch):
        batch = self.reshape_tensor(batch)
        
        max_generation_len = 30

        beam_size = self.beam_size
        batch_size = self.get_batch_size(batch)
        seq_logprob = ToCuda(torch.zeros((batch_size, 1, 1)))
        log_probs = []
        selected_words = None
        seq_mask = ToCuda(torch.ones((batch_size, beam_size, 1)))

        state = None
        cache = {'key': {}, 'value': {}, 'attn_masks': None}
        outputs = []
        for t in range(max_generation_len):
            cur_beam_size = 1 if t == 0 else beam_size

            word_logits = self.get_logits(batch, state)
            word_logprob = F.log_softmax(word_logits, dim=1)

            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            candidate_logprob = seq_logprob + word_logprob

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != self.tokenizer.sep_token).float().unsqueeze(-1)
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

    def expand_tensor(self, tensor, size, dim=1):
        if size == 1 or tensor is None:
            return tensor
        tensor = tensor.unsqueeze(dim)
        tensor = tensor.expand(list(tensor.shape[:dim]) + [size] + list(tensor.shape[dim + 1:])).contiguous()
        tensor = tensor.view(list(tensor.shape[:dim - 1]) + [-1] + list(tensor.shape[dim + 1:]))
        return tensor

    def get_batch_size(self, batch):
        return batch['imgs'].shape[0]

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


class ModelWrapper(nn.Module):
    
    def __init__(self, module):
        super().__init__()
        self.module = module
