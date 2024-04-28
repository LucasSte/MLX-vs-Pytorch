# This file will download the PTB corpus dataset https://paperswithcode.com/dataset/penn-treebank
import pathlib
import os
from urllib import request
import numpy as np
import itertools


# Loading function copied from: https://github.com/ml-explore/mlx-examples/blob/main/transformer_lm/datasets.py
def load_ptb():
    current_dir = str(pathlib.Path(__file__).parent.resolve())
    save_dir = current_dir + "/tmp/ptb"

    contents = [
        "ptb.train.txt",
        "ptb.valid.txt",
        "ptb.test.txt",
    ]

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        base_url = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/"
        for file_name in contents:
            save_path = os.path.join(save_dir, file_name)
            if not os.path.exists(save_path):
                request.urlretrieve(base_url + file_name, save_path)

    # Loading
    with open(os.path.join(save_dir, contents[0]), "r") as f:
        vocab = set(t for l in f.readlines() for t in l.strip().split(" "))
    eos = "<eos>"
    vocab.add(eos)
    vocab = {v: i for i, v in enumerate(vocab)}

    def to_array(dataset):
        with open(os.path.join(save_dir, dataset), "r") as f:
            lines = (l.strip().split(" ") for l in f.readlines())
        return np.array(
            [vocab[w] for line in lines for w in itertools.chain(line, [eos])],
            dtype=np.uint32,
        )

    datasets = [to_array(fn) for fn in contents]
    return vocab, *datasets
