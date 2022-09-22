from model.model import QuestionerGenerator, ModelWrapper
from data.dataset import Tokenizer, build_dataset
from torch.utils.data import DataLoader
from utils import ToCuda
import torch
import json


# w/o distributed running
def main():
    with torch.no_grad():
        with open('images.json') as f:
            all_images = json.load(f)
        # tokenizer
        tokenizer = Tokenizer()
        # build and load model
        model = QuestionerGenerator(tokenizer=tokenizer)
        model = ModelWrapper(model)
        checkpoint = torch.load('./log/test/checkpoint/checkpoint.pt', map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.eval()
        del checkpoint
        model: QuestionerGenerator = ToCuda(model.module)
        val_dataset = DataLoader(build_dataset('answer', 'val', tokenizer, True, 'all'), batch_size=1, shuffle=True)
        
        max_iter = 20
        for i, batch in enumerate(val_dataset):
            tip = torch.flatten(batch[0]['tips'], start_dim=0, end_dim=1)
            target = torch.flatten(batch[1], start_dim=0, end_dim=1)
            
            img = []
            for item in zip(*batch[2]):
                img.extend(list(item))
            if img[0] in all_images['images']:
                max_iter += 1
                continue

            prediction = model(batch[0])
        
            detach = lambda x: x.detach().cpu().tolist()

            for answer, question, output, image in zip(convert_items(detach(tip), tokenizer),
                                                       convert_items(detach(target), tokenizer),
                                                       convert_items(detach(prediction), tokenizer),
                                                       img):
                append_to_file('result.txt', ('Image: {0} - Answer: {1} - Question: {2} - Output: {3}\n'.format(
                    image,
                    answer,
                    question,
                    output
                )))
            
            items = map(lambda x: int(x), list(set(img)))
            all_images['images'].extend(items)
            print(list(items))
            with open('images.json', 'w') as f:
                json.dump(all_images, f, indent=4)

            if i >= max_iter - 1:
                break


def append_to_file(path, content):
    with open(path, 'a') as f:
        f.write(content)


def convert_items(items, tokenizer: Tokenizer):
    results = []
    for item in items:
        results.append(' '.join(tokenizer.tokenizer.convert_ids_to_tokens(item[(0 if item[0] != 101 else 1):item.index(102)])))
    return results


if __name__ == '__main__':
    main()
