from vqa_utils.vqaTools.vqa import VQA
from utils import parse_yaml, default_config_path


def main():
    config = parse_yaml(default_config_path)

    question_type = set()
    answer_type = set()
    
    def count(annotation, question):
        vqa = VQA(annotation, question)
        ids = list(set(vqa.getImgIds()))
        for _id in ids:
            q_ids = vqa.getQuesIds(_id, ansTypes=['number', 'other'])
            for q_id in q_ids:
                item = vqa.loadQA(q_id)[0]
                question_type.add(item['question_type'])
                answer_type.add(item['answer_type'])

    count(config['train']['annotation'], config['train']['question'])
    count(config['val']['annotation'], config['val']['question'])
    
    print(question_type)
    print(answer_type)


if __name__ == '__main__':
    main()
