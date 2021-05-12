"""
"""

import copy
import json
from typing import Any, Dict, List, Tuple
from itertools import chain
from collections import defaultdict

from nmnlp.core import DataSet

EXC_AID = {
    'none': set(),
    'bad': {24, 20, 42, 18, 11, 13, 46, 38, 35, 12, 37, 5, 16, 21, 14, 33},
    'small': set(i for i in range(48) if i not in {
        34, 2, 10, 8, 25, 23, 28, 20, 3, 44, 20, 12, 14}),
    'middle': set(i for i in range(48) if i not in {
        34, 2, 36, 8, 25, 23, 28, 44, 45, 20, 13, 21, 14}),
}
# 3, 27, 44, 32, 45, 29, 17,


def read_lines(path, sep=" "):
    # print('reading ', path)
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
        distant_gold=False, crowd_test=False, exclude='none', tokenizer=None,
        seed=123
    ) -> Dict[str, 'CoNLL03Crowd']:
        test_set = cls(cls.single_label('data/test.bio', tokenizer))
        dev_set = cls(cls.single_label('data/dev.bio', tokenizer))

        if 'answers' in name:
            train = cls.crowd_label(data_dir + name, tokenizer, exclude)
        else:
            train = cls.single_label(data_dir + name, tokenizer)

        if extra_gold is not None:
            if distant_gold:
                name = 'rest'
            else:
                name = 'gold'
            name += f'-{str(seed)}'
            sampled = cls.single_label(f"{data_dir}/{name}-{extra_gold}.txt", tokenizer)
            if only_gold:
                train = sampled
            elif replace:
                train = cls.replace_gold(train, sampled)
            else:
                train.extend(sampled)

        out = dict(train=cls(train), dev=dev_set, test=test_set)
        if crowd_test:
            kept = cls.crowd_label(data_dir + 'answers-15.txt', tokenizer, exclude)
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
        print(f"    Read {len(data)} instances from <{path}>")
        return data

    @classmethod
    def crowd_label(cls, path, tokenizer, exclude) -> List[Dict[str, Any]]:
        data, bad_ids = list(), EXC_AID[exclude]
        for tid, lines in enumerate(read_lines(path)):
            words = [li[0] for li in lines]
            for i in range(1, len(lines[0])):
                if i in bad_ids:
                    continue
                tags = [li[i] for li in lines]
                if len(set(tags)) > 1:
                    aw, tags = word_piece_tokenzie(
                        copy.deepcopy(words), tags, tokenizer)
                    data.append(cls.to_instance(aw, tags, tid, i))
        print(f"    Read {len(data)} instances from <{path}>")
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


def read_PICO(name='crowdsourcing', root='./PICO-data'):
    worker_c, doc_c = defaultdict(int), defaultdict(int)
    with open(f"{root}/annotations/train/PICO-annos-{name}.json", mode='r') as f:
        for line in f:
            ins = json.loads(line)
            doc_c[ins['docid']] = len(ins['Participants'])
            for k, v in ins['Participants'].items():
                worker_c[k] += len(v)

    return worker_c, doc_c


def read_AURC(root='./AURC'):
    import re
    regex = re.compile(r"\([0-9]+,[0-9]+\)")
    rec, rep = re.compile(r"con"), re.compile(r"pro")
    worker_c, doc_c = defaultdict(int), defaultdict(int)
    wc_c, wp_c = defaultdict(int), defaultdict(int)
    with open(f"{root}/data/AURC_DATA.tsv", mode='r') as f:
        _ = f.readline().strip().split('\t')
        for i, line in enumerate(f.readlines()):
            cols = line.strip().split('\t')
            num = 0
            for j, c in enumerate(cols[10:]):
                if 'false' in c:
                    n = len(regex.findall(c))
                    nc = len(rec.findall(c))
                    np = len(rep.findall(c))
                    num += 1
                    worker_c[j] += n
                    wc_c[j] += nc
                    wp_c[j] += np
            doc_c[i] = num

    return worker_c, doc_c, (wc_c, wp_c)


def read_NER(root='./data'):
    bad_ids = EXC_AID['small']
    worker_c, doc_c = defaultdict(int), defaultdict(int)
    labels_c = {'PER': defaultdict(int), 'LOC': defaultdict(int), 'ORG': defaultdict(int), 'MISC': defaultdict(int)}
    data = list(read_lines(root + '/answers.txt'))
    for i, lines in enumerate(data):
        num = 0
        for j in range(1, len(lines[0])):
            if j in bad_ids:
                continue
            nt = 0
            for li in lines:
                tag = li[j]
                if tag.startswith('B'):
                    worker_c[j] += 1
                    label = tag.split('-')[1]
                    labels_c[label][j] += 1
                    nt += 1
            if nt > 0:
                num += 1
        if num > 0:
            doc_c[i] = num

    def write_lines(data, path, sep=' '):
        with open(path, mode='w', encoding='UTF-8') as file:
            for ins in data:
                for line in ins:
                    file.write(sep.join(line) + '\n')
                file.write('\n')
        print('write to ', path)

    # save_data = [data[k] for k, v in doc_c.items() if v > 2]
    # write_lines(save_data, 'data/one-third/answers.txt')
    # gold = list(read_lines(root + '/ground_truth.txt'))
    # save_gold = [gold[k] for k, v in doc_c.items() if v > 2]
    # write_lines(save_gold, 'data/1_3-ground_truth.txt')
    # mv = list(read_lines(root + '/mv.txt'))
    # save_mv = [mv[k] for k, v in doc_c.items() if v > 2]
    # write_lines(save_mv, 'data/one-third/mv.txt')

    # save_data = [data[k] for k, v in doc_c.items() if v > 1]
    # write_lines(save_data, 'data/small/answers.txt')
    # gold = list(read_lines(root + '/ground_truth.txt'))
    # save_gold = [gold[k] for k, v in doc_c.items() if v > 1]
    # write_lines(save_gold, 'data/small/ground_truth.txt')
    # mv = list(read_lines(root + '/mv.txt'))
    # save_mv = [mv[k] for k, v in doc_c.items() if v > 1]
    # write_lines(save_mv, 'data/small/mv.txt')

    # 4888 > 0, 4188 > 1, 3457 > 2 每句有效标注人

    return worker_c, doc_c, labels_c


def main():
    # read_PICO()
    # read_AURC()
    read_NER()
    # CoNLL03Crowd.build('dev/data/conll03-crowd/', 'ground_truth.txt')


if __name__ == "__main__":
    main()
