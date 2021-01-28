"""
"""

import copy
import random
import pickle
from typing import Any, Dict, List, Tuple
# from difflib import SequenceMatcher
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
                else:
                    print(cols)


class CoNLL03Crowd(DataSet):
    index_fields = ('words', 'tags')
    label_set = set()

    @classmethod
    def build(
        cls, data_dir, name, extra_gold=None, only_gold=False, replace=False,
        distant_gold=False, reserve_crowd=False, tokenizer=None
    ) -> Dict[str, 'CoNLL03Crowd']:
        test_set = cls(cls.single_label(data_dir + 'test.bio', tokenizer))
        dev_set = cls(cls.single_label(data_dir + 'dev.bio', tokenizer))

        if 'answers' in name:
            train = cls.crowd_label(data_dir + name, tokenizer)
        else:
            train = cls.single_label(data_dir + name, tokenizer)

        if extra_gold is not None:
            if distant_gold:
                with open('dev/data/distant.pkl', mode='rb') as f:
                    gold = pickle.load(f)
                    print('loaded distant train')
            else:
                gold = cls.single_label(data_dir + 'ground_truth.txt', tokenizer)
                # conll = cls.single_label(data_dir + 'train.bio', tokenizer)
                # distant, matched = set(), set()
                # for i, c in enumerate(conll):
                #     text = ''.join(c['text'])
                #     for j, g in enumerate(gold):
                #         if j in matched:
                #             continue
                #         sim = SequenceMatcher(None, text, ''.join(g['text'])).quick_ratio()
                #         if sim > 0.98:
                #             matched.add(j)
                #             break
                #     else:
                #         distant.add(i)
                # gold = [c for i, c in enumerate(conll) if i in distant]
            if extra_gold <= 1:
                extra_gold = 5985 * extra_gold
            sampled = random.sample(gold, int(extra_gold))
            print(f"--- sampled {int(extra_gold)} gold instances.")
            if only_gold:
                train = sampled
            elif replace:
                train = cls.replace_gold(train, sampled)
            else:
                train.extend(sampled)

        if reserve_crowd:
            out = dict(dev=dev_set, test=test_set)
            kept = list()
            for i in range(len(train)):
                if random.uniform(0, 1) >= 0.85:
                    kept.append(i)
            crowd = [ins for i, ins in enumerate(train) if i not in kept]
            kept = [ins for i, ins in enumerate(train) if i in kept]
            out = dict(train=cls(crowd), dev=dev_set, test=test_set, kept=cls(kept))
        else:
            out = dict(train=cls(train), dev=dev_set, test=test_set)

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
    def crowd_label(cls, path, tokenizer) -> List[Dict[str, Any]]:
        data = list()
        for tid, lines in enumerate(read_lines(path)):
            words = [li[0] for li in lines]
            for i in range(1, len(lines[0])):
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
