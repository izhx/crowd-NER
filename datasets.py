"""
"""

import copy
from typing import Dict
from itertools import chain

from nmnlp.core import DataSet


def read_lines(path, sep=" "):
    with open(path, mode='r', encoding='UTF-8') as conllu_file:
        sentence = list()
        for line in chain(conllu_file, [""]):
            line = line.strip()
            if not line and sentence:
                yield sentence
                sentence = list()
            else:
                cols = line.split(sep)
                if len(cols) > 1:
                    sentence.append(cols)
                # else:
                #     print(cols)


class CoNLL03Crowd(DataSet):
    index_fields = ('words', 'tags')
    label_set = set()

    @classmethod
    def build(cls, data_dir, name) -> Dict[str, 'CoNLL03Crowd']:
        dev_set = cls.single_label(data_dir + 'testset.txt')
        if 'answers' in name:
            train_set = cls.crowd_label(data_dir + name)
        else:
            train_set = cls.single_label(data_dir + name)

        return dict(train=train_set, dev=dev_set)

    @classmethod
    def single_label(cls, path) -> 'CoNLL03Crowd':
        data = list()
        for lines in read_lines(path):
            data.append(dict(
                words=[li[0] for li in lines], tags=[li[1] for li in lines],
                text=[li[0] for li in lines]))
        return cls(data)

    @classmethod
    def crowd_label(cls, path) -> 'CoNLL03Crowd':
        data = list()
        for tid, lines in enumerate(read_lines(path)):
            words = [li[0] for li in lines]
            for i in range(1, len(lines[0])):
                tags = [li[i] for li in lines]
                if len(set(tags)) > 1:
                    data.append(dict(
                        words=copy.deepcopy(words), tags=tags, aid=i-1,
                        tid=tid, text=copy.deepcopy(words)))
        return cls(data)


def main():
    CoNLL03Crowd.build('dev/data/conll03-crowd/', 'ground_truth.txt')


if __name__ == "__main__":
    main()
