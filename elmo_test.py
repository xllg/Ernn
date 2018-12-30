from allennlp.commands.elmo import ElmoEmbedder
import numpy as np
import json

options_file = "../data/datasets/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "../data/datasets/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

train_file = "../data/datasets/SQuAD-v1.1-train-processed-corenlp.txt"

def load_data(filename, skip_no_answer=False):
    # Load JSON lines
    with open(filename) as f:
        examples = [json.loads(line) for line in f]
    for ex in examples:
        ex['question'] = [w.lower() for w in ex['question']]
        ex['document'] = [w.lower() for w in ex['document']]
    # Skip unparsed (start/end) examples
    if skip_no_answer:
        examples = [ex for ex in examples if len(ex['answers']) > 0]
    return examples

if __name__ == '__main__':
    # example = load_data(train_file)

    elmo = ElmoEmbedder(options_file, weight_file)

    # use batch_to_ids to convert sentences to character ids
    sentences = [['I', 'like', '(', 'apple', ')', '.'], ['I', 'like', '(', 'apple', ')', 'phone']]
    embeddings, _ = elmo.batch_to_embeddings(sentences)

    love1 = embeddings[0][2][3]

    # love1 = elmo_embedding[0][2][3]
    # love2 = elmo_embedding[1][2][3]
    # vector_a = np.mat(love1)
    # vector_b = np.mat(love2)
    # num = float(vector_a * vector_b.T)
    # denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    # cos = num / denom
    # sim = 0.5 + 0.5 * cos

    # print(sim)


