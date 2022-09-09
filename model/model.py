import torch.nn as nn
import torch
from .clip import build_model
from .transformer import GELU
from .bert import BertModel, BertConfig
from apex.normalization.fused_layer_norm import FusedLayerNorm
import os
from utils import default_config_path, parse_yaml
import copy


class Questioner(nn.Module):

    def __init__(self, mask_token, config_path=default_config_path):
        super().__init__()
        self.video_dim = 1024
        self.multimodal_dim = 768
        self.build_model_structure(config_path=config_path)
        self.masker = TokenMasker(mask_token=mask_token, range_start=106, range_end=30522)

    def forward(self, batch):
        img, tip, target = batch
        if len(img.size()) > 5:
            pass
        if len(tip.size()) > 2:
            pass
        
        # random mask to the target(also called 'question')
        target_tokens, labels_qa = self.masker(target, 0.99)
        # get encoded and embedded img features
        img_input = self.forward_video(img)
        img_input = self.get_video_multimodal_embedding(img_input)
        
        
    
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
        video = video + self.video_type_embeddings
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

    def load_pretrained_weights(self, config_path=default_config_path):
        """Load pretrained bert and clip respectively using config file.

        Args:
            config_path (str, optional): config path. Defaults to default_config_path.
        """
        config = parse_yaml(config_path)
        pretrained_path = config['pretrained_path']
        # load clip and bert weights
        self.clip.load_state_dict(torch.load(os.path.join(pretrained_path, config['clip_pretrained_path'])))
        self.bert.load_state_dict(torch.load(os.path.join(pretrained_path, config['bert_pretrained_path'])))
        self.cls_head.load_state_dict(torch.load(os.path.join(pretrained_path, config['cls_head_pretrained_path'])))
        self.video_feature_adapter.load_state_dict(torch.load(os.path.join(pretrained_path, config['video_feature_adapter_pretrained_path'])))
        self.video_frame_embedding = nn.Parameter(torch.load(os.path.join(pretrained_path, config['video_frame_embedding_pretrained_path'])))
        self.video_type_embedding = nn.Parameter(torch.load(os.path.join(pretrained_path, config['video_type_embedding_pretrained_path'])))

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

        ### generate indicator first:
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
                    prob = random.random()   #### e-6 too much time
                    if prob < 0.8:
                        tokens[i][j] = self.mask_token  ### e-6 have no idea why too much
                    elif prob < 0.9: 
                        tokens[i][j] = random.choice(list(range(*self.range)))
                    #tokens[i][j] = self.mask_token
                    labels[i][j] = src_token

        tokens = torch.from_numpy(tokens).long().cuda()
        labels = torch.from_numpy(labels).long().cuda()

        return tokens, labels
