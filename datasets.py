"""
"""

import copy
from typing import Dict, List, Tuple
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
            elif line.startswith("-DOCSTART-"):
                continue
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
    def build(cls, data_dir, name, tokenizer=None) -> Dict[str, 'CoNLL03Crowd']:
        test_set = cls.single_label(data_dir + 'test.bio', tokenizer)
        dev_set = cls.single_label(data_dir + 'dev.bio', tokenizer)
        if 'answers' in name:
            train_set = cls.crowd_label(data_dir + name, tokenizer)
        else:
            train_set = cls.single_label(data_dir + name, tokenizer)

        return dict(train=train_set, dev=dev_set, test=test_set)

    @staticmethod
    def to_instance(words, tags, tid, aid=None):
        ins = dict(words=words, tags=tags, tid=tid, text=copy.deepcopy(words))
        if aid is not None:
            ins.update(aid=aid)
        return ins

    @classmethod
    def single_label(cls, path, tokenizer) -> 'CoNLL03Crowd':
        data = list()
        for tid, lines in enumerate(read_lines(path)):
            words, tags = word_piece_tokenzie(
                [li[0] for li in lines], [li[-1] for li in lines], tokenizer)
            data.append(cls.to_instance(words, tags, tid))
        return cls(data)

    @classmethod
    def crowd_label(cls, path, tokenizer) -> 'CoNLL03Crowd':
        data = list()
        for tid, lines in enumerate(read_lines(path)):
            words = [li[0] for li in lines]
            for i in range(1, len(lines[0])):
                tags = [li[i] for li in lines]
                if len(set(tags)) > 1:
                    aw, tags = word_piece_tokenzie(
                        copy.deepcopy(words), tags, tokenizer)
                    data.append(cls.to_instance(aw, tags, tid, i-1))
        return cls(data)


def word_piece_tokenzie(texts, tags, tokenizer) -> Tuple[List[str], List[str]]:
    if tokenizer is None:
        return texts, tags

    words, expanded_tags = ["[CLS]"], ["O"]
    for text, tag in zip(texts, tags):
        pieces = tokenizer.tokenize(text)
        if tag.startswith("B-"):
            i_tag = tag.replace("B-", "I-")
            piece_tags = [tag] + [i_tag for _ in range(len(pieces) - 1)]
        else:
            piece_tags = [tag for _ in pieces]
        words.extend(pieces)
        expanded_tags.extend(piece_tags)
    else:
        words.append("[SEP]")
        expanded_tags.append("O")

    return words, expanded_tags


def main():
    CoNLL03Crowd.build('dev/data/conll03-crowd/', 'ground_truth.txt')


if __name__ == "__main__":
    main()
