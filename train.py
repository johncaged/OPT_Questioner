from torch_lib import Proxy
from model.model import Questioner
import os
from data.dataset import VQADataset, Tokenizer, TrainImageProcessor, QuestionAnswerProcessor, QuestionCaptionProcessor


# QAs = VQADataset(
#     TrainImageProcessor(),
#     QuestionAnswerProcessor('./dataset/VQA/v2_OpenEnded_mscoco_train2014_questions.json', './dataset/VQA/v2_mscoco_train2014_annotations.json', Tokenizer(), mode='all'),
#     './dataset/COCO/train2014',
#     'COCO_train2014_'
# )


model = Questioner(Tokenizer().mask_token)
model.load_pretrained_weights()
proxy = Proxy(model)
print('load finished')
