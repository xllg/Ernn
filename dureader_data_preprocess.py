#-*- coding:utf-8 -*-
'''
@author: xllg
@project: AttReader
@file: dureader_data_preprocess.py
@time: 19-4-30 下午6:57
'''

import json
import argparse
import os

def process_dataset(in_path):
    """Load json file and store fields separately."""
    processd_dataset = []
    processd_corpus = []
    with open(in_path) as f:
        for line in f:
            sample = json.loads(line.strip())
            if len(sample['answer_spans']) == 0:
                continue
            if sample['answer_spans'][0][1] >= 600:
                continue
            # answer = sample['answer_spans'][0]

            # if (answer[1] - answer[0] + 1) > 200:
            #     s = answer[0]
            #     e = answer[0] + 199
            # else:
            #     s = answer[0]
            #     e = answer[1]

            # sample['answer_spans'] = [[s, e]]

            id = sample['question_id']
            question = sample['segmented_question'][:60]
            # 寻找最相关的文本片段
            answer_doc = sample['answer_docs'][0]
            if answer_doc < len(sample['documents']):
                doc = sample['documents'][answer_doc]
            else:
                continue
                # doc = sample['documents'][-1]
            most_related_para = doc['most_related_para']
            document = doc['segmented_paragraphs'][most_related_para][:600]
            answers = sample['answer_spans']
            if answers[0][1] >= len(document):
                continue
                # print(id)
                # answers[0][1] = len(document) - 1

            ex_dataset = {
                'id': id,
                'question': question,
                'document': document,
                'answers': answers,
            }
            ex_corpus = (' '.join(document))
            processd_dataset.append(ex_dataset)
            processd_corpus.append(ex_corpus)
    return processd_dataset, processd_corpus

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Path to SQuAD data directory',
                    default='/home/xllg/PycharmProjects/AttReader/data/dureaderpre')
parser.add_argument('--out_dir', type=str, help='Path to output file dir',
                    default='/home/xllg/PycharmProjects/AttReader/data/dureaderpre')
parser.add_argument('--split', type=str, help='Filename for train/dev/test split',
                    default='trainset/zhidao.train')
args = parser.parse_args()

in_file = os.path.join(args.data_dir, args.split + '.json')
print('Processing dataset!')
dataset, corpus = process_dataset(in_file)

dataset_out_file = os.path.join(args.out_dir, '%s-processd_dataset.txt' % (args.split))
corpus_out_file = os.path.join(args.out_dir, '%s-processd_corpus.txt' % (args.split))

print('Writing dataset!')
with open(dataset_out_file, 'w') as fo:
    for item in dataset:
        fo.write(json.dumps(item, ensure_ascii=False) + '\n')  # dumps方法默认使用ASCII对中文进行编码，将其关闭

# print('Writing corpus!')
# with open(corpus_out_file, 'w') as fo:
#     for item in corpus:
#         fo.write(item + '\n')

print('Done!')