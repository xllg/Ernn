import numpy as np
import os
from pathlib import PosixPath

ErnnReader_DATA = os.path.join(PosixPath(__file__).absolute().parents[1].as_posix(), 'data')
EMBED_DIR = os.path.join(ErnnReader_DATA, 'embeddings')
file_path = os.path.join(EMBED_DIR, "glove.840B.300d.txt")

vectors = {}
with open(file_path, 'r') as f:
    for line in f:
        line_split = line.rstrip().split(' ')
        # if line_split.__len__()==300:
        #     print(line_split)
        vec = np.array(line_split[1:], dtype=float)
        word = line_split[0]

        for char in word:
            if ord(char) < 128:
                if char in vectors:
                    vectors[char] = (vectors[char][0] + vec,
                                     vectors[char][1] + 1)
                else:
                    vectors[char] = (vec, 1)

base_name = os.path.join(EMBED_DIR, "glove.840B.300d-char.txt")
with open(base_name, 'w') as f2:
    for word in vectors:
        avg_vector = np.round(
            (vectors[word][0] / vectors[word][1]), 6).tolist()
        f2.write(word + " " + " ".join(str(x) for x in avg_vector) + "\n")