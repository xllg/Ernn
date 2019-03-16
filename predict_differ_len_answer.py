#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to make and save model predictions on an input dataset."""

import os
import time
import torch
import argparse
import logging
import json

from tqdm import tqdm
from predictor import Predictor

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='/home/xllg/PycharmProjects/AttReader/data/datasets/SQuAD-v1.1-dev.json',
                    help='SQuAD-like dataset to evaluate on')
parser.add_argument('--model', type=str, default=None,
                    help='Path to model to use')
parser.add_argument('--embedding-file', type=str, default=None,
                    help=('Expand dictionary to use all pretrained '
                          'embeddings in this file.'))
parser.add_argument('--out-dir', type=str, default='/home/xllg/PycharmProjects/AttReader/data/drqa_up_predict',
                    help=('Directory to write prediction file to '
                          '(<dataset>-<model>.preds)'))
parser.add_argument('--tokenizer', type=str, default=None,
                    help=("String option specifying tokenizer type to use "
                          "(e.g. 'corenlp')"))
parser.add_argument('--num-workers', type=int, default=None,
                    help='Number of CPU processes (for tokenizing, etc)')
parser.add_argument('--no-cuda', action='store_true',
                    help='Use CPU only')
parser.add_argument('--gpu', type=int, default=-1,
                    help='Specify GPU device id to use')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Example batching size')
parser.add_argument('--top-n', type=int, default=1,
                    help='Store top N predicted spans per example')
parser.add_argument('--official', action='store_true', default=True,
                    help='Only store single top span instead of top N list')
args = parser.parse_args()
t0 = time.time()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpu)
    logger.info('CUDA enabled (GPU %d)' % args.gpu)
else:
    logger.info('Running on CPU only.')

predictor = Predictor(
    args.model,
    args.tokenizer,
    args.embedding_file,
    args.num_workers,
)
if args.cuda:
    predictor.cuda()


# ------------------------------------------------------------------------------
# Read in dataset and make predictions.
# ------------------------------------------------------------------------------

# predict different length of answer
ans_len=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29]
examples_all = [[] for i in range(len(ans_len))]
qids = [[] for i in range(len(ans_len))]
with open(args.dataset) as f:
    data = json.load(f)['data']
    for article in data:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                ans = qa['answers'][0]['text'].split(' ')  # ans_len
                # if 3*(ans_len-1)<= len(ans) <= 3 * ans_len:  #
                id = 27 if len(ans) == 29 else len(ans)
                qids[id-1].append(qa['id'])
                examples_all[id-1].append((context, qa['question']))

for num, examples in enumerate(examples_all):
    results = {}
    qid = qids[num]
    for i in tqdm(range(0, len(examples), args.batch_size)):
        predictions = predictor.predict_batch(
            examples[i:i + args.batch_size], top_n=args.top_n
        )
        for j in range(len(predictions)):
            # Official eval expects just a qid --> span
            if args.official:
                results[qid[i + j]] = predictions[j][0][0]

            # Otherwise we store top N and scores for debugging.
            else:
                results[qid[i + j]] = [(p[0], float(p[1])) for p in predictions[j]]

    # model = os.path.splitext(os.path.basename(args.model or 'default'))[0]
    # basename = os.path.splitext(os.path.basename(args.dataset))[0]
    # outfile = os.path.join(args.out_dir, basename + '-' + model + '.preds')
    outfile = os.path.join(args.out_dir, 'ans-len'+str(ans_len[num])+'.preds')

    logger.info('Writing results to %s' % outfile)
    with open(outfile, 'w') as f:
        json.dump(results, f)

    logger.info('Total time: %.2f' % (time.time() - t0))
