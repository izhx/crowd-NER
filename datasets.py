"""
"""

import copy
from typing import Any, Dict, List, Tuple
from itertools import chain

from nmnlp.core import DataSet

BAD_AID = (24, 20, 42, 18, 11, 13, 46, 38, 35, 12, 37, 5, 16, 21, 14, 33)
# 3, 27, 44, 32, 45, 29, 17,


def read_lines(path, sep=" "):
    print('reading ', path)
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
                else:
                    print(cols)


class CoNLL03Crowd(DataSet):
    index_fields = ('words', 'tags')
    label_set = set()

    @classmethod
    def build(
        cls, data_dir, name, extra_gold=None, only_gold=False, replace=False,
        distant_gold=False, crowd_test=False, exclude_bad=False, tokenizer=None
    ) -> Dict[str, 'CoNLL03Crowd']:
        test_set = cls(cls.single_label(data_dir + 'test.bio', tokenizer))
        dev_set = cls(cls.single_label(data_dir + 'dev.bio', tokenizer))

        if 'answers' in name:
            train = cls.crowd_label(data_dir + name, tokenizer, exclude_bad)
        else:
            train = cls.single_label(data_dir + name, tokenizer)

        if extra_gold is not None:
            if distant_gold:
                name = 'rest'
            else:
                name = 'gold'
            sampled = cls.single_label(f"{data_dir}/{name}-{extra_gold}.txt", tokenizer)
            if only_gold:
                train = sampled
            elif replace:
                train = cls.replace_gold(train, sampled)
            else:
                train.extend(sampled)

        out = dict(train=cls(train), dev=dev_set, test=test_set)
        if crowd_test:
            kept = cls.crowd_label(data_dir + 'answers-15.txt', tokenizer, exclude_bad)
            conll_dev = out['dev']
            out.update(dev=cls(kept), conll_dev=conll_dev)

        return out

    @staticmethod
    def to_instance(words, tags, tid, aid=0):
        ins = dict(words=words, tags=tags, aid=aid, tid=tid, text=copy.deepcopy(words))
        return ins

    @classmethod
    def single_label(cls, path, tokenizer) -> List[Dict[str, Any]]:
        data = list()
        for tid, lines in enumerate(read_lines(path)):
            words, tags = word_piece_tokenzie(
                [li[0] for li in lines], [li[-1] for li in lines], tokenizer)
            data.append(cls.to_instance(words, tags, tid))
        return data

    @classmethod
    def crowd_label(cls, path, tokenizer, exclude_bad) -> List[Dict[str, Any]]:
        data = list()
        for tid, lines in enumerate(read_lines(path)):
            words = [li[0] for li in lines]
            for i in range(1, len(lines[0])):
                if exclude_bad and i in BAD_AID:
                    continue
                tags = [li[i] for li in lines]
                if len(set(tags)) > 1:
                    aw, tags = word_piece_tokenzie(
                        copy.deepcopy(words), tags, tokenizer)
                    data.append(cls.to_instance(aw, tags, tid, i))
        return data

    @staticmethod
    def replace_gold(train, sampled):
        matched = set()
        for ins in train:
            for g in range(len(sampled)):
                if g in matched:
                    continue
                gold = sampled[g]
                if ins['text'] == gold['text']:
                    ins['tags'] = gold['tags']
                    matched.add(g)
                    break
        return train


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
