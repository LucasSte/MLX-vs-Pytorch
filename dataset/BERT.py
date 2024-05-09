import gzip
import pathlib
import os
from urllib import request


def load_nli():
    current_dir = pathlib.Path(__file__).parent.resolve()
    parent_save_dir = os.path.join(current_dir, "tmp")
    save_dir = os.path.join(parent_save_dir, "ptb")

    if not os.path.exists(parent_save_dir):
        os.mkdir(parent_save_dir)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        url = "https://sbert.net/datasets/AllNLI.tsv.gz"
        save_path = os.path.join(save_dir, "AllNLI.tsv.gz")
        if not os.path.exists(save_path):
            request.urlretrieve(url, save_path)

    test = [[], [], []]
    dev = [[], [], []]
    train = [[], [], []]

    # TODO: Should this be a torch tensor or an mx array?
    mp = {"contradiction": [1, 0, 0], "neutral": [0, 1, 0], "entailment": [0, 0, 1]}

    file_name = os.path.join(save_dir, "AllNLI.tsv.gz")
    with gzip.open(file_name, "rb") as file:
        train_items = 0
        for line in file.readlines():
            line_as_string = line.decode("utf-8")
            items = line_as_string.strip("\n\r").split("\t")
            # As we do not need more than 50.000 items, we can ignore the rest
            if items[0] == "train" and train_items <= 50000:
                train[0].append(items[3])
                train[1].append(items[4])
                train[2].append(mp[items[5]])
                train_items += 1
            # I am ignoring the test set to save RAM
            # elif items[0] == 'test':
            #   test[0].append(items[3])
            #   test[1].append(items[4])
            #   test[2].append(mp[items[5]])
            elif items[0] == "dev":
                dev[0].append(items[3])
                dev[1].append(items[4])
                dev[2].append(mp[items[5]])

    samples = {"test": test, "dev": dev, "train": train}

    return samples
