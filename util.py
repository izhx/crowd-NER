
import random
import pickle
from typing import List, Tuple
# from difflib import SequenceMatcher
from itertools import chain

from nmnlp.core import Vocabulary


def allowed_transition(vocab: Vocabulary, namespace='tags') -> List[Tuple]:
    def idx(token: str) -> int:
        return vocab.index_of(token, namespace)

    allowed, keys = list(), vocab.token_to_index[namespace].keys()
    for i in keys:
        for j in keys:
            if i == "O" and j.startswith("I-"):
                continue
            if i.startswith("B-") and j.startswith("I-") and i.split('-')[1] != j.split('-')[1]:
                continue
            if i.startswith("I-") and j.startswith("I-") and i.split('-')[1] != j.split('-')[1]:
                continue
            allowed.append((idx(i), idx(j)))

    return allowed


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


def write_lines(data, path, sep=' '):
    with open(path, mode='w', encoding='UTF-8') as file:
        for ins in data:
            for line in ins:
                file.write(sep.join(line) + '\n')
            file.write('\n')
    print('write to ', path)


def sample_data():
    DIR = 'dev/data/conll03-crowd/'
    # crowd = list(read_lines(DIR + 'answers.txt'))
    # train, test = list(), list()
    # for line in crowd:
    #     if random.uniform(0, 1) >= 0.85:
    #         test.append(line)
    #     else:
    #         train.append(line)
    # write_lines(train, DIR + "answers-85.txt")
    # write_lines(test, DIR + "answers-15.txt")

    gold = list(read_lines(DIR + 'ground_truth.txt'))
    # conll = list(read_lines(DIR + 'train.bio'))
    # distant, matched = set(), set()
    # for i, ins in enumerate(conll):
    #     text = ''.join(c[0] for c in ins)
    #     for j, ing in enumerate(gold):
    #         if j in matched:
    #             continue
    #         sim = SequenceMatcher(None, text, ''.join(c[0] for c in ing)).quick_ratio()
    #         if sim > 0.98:
    #             matched.add(j)
    #             break
    #     else:
    #         distant.add(i)
    # conll_rest = [c for i, c in enumerate(conll) if i in distant]

    def filter(lines, tags=2, length=8) -> bool:
        if len(lines) < length:
            return False
        if sum(1 if 'B-' in i[1] else 0 for i in lines) < tags:
            return False
        return True

    with open('dev/data/rest.pkl', mode='rb') as f:
        conll_rest = pickle.load(f)

    def multi(data, name):
        good = set(i for i, lines in enumerate(data) if filter(lines, 3))
        print('good: ', len(good))
        p1 = set(random.sample(good, 60))
        p1d = [d for i, d in enumerate(data) if i in p1]
        print('prop: 0.01, num: ', len(p1d))
        write_lines(p1d, f"{DIR}{name}-0.01.txt")
        left = good.difference(p1)
        print('1 left: ', len(left))
        p5 = set(random.sample(left, 240))
        p5d = p1d + [d for i, d in enumerate(data) if i in p5]
        print('prop: 0.05, num: ', len(p5d))
        write_lines(p5d, f"{DIR}{name}-0.05.txt")
        left = left.difference(p5)
        print('5 left: ', len(left))

        two = set(i for i, lines in enumerate(data) if filter(lines, 2))
        two = two.difference(good)
        need = 1197 - len(left)
        if need > 0:
            print('need: ', need)
            p25 = left.union(set(random.sample(two, need)))
        else:
            p25 = set(random.sample(left, 1197))
            left = left.difference(p25)
            print('25 left: ', len(left))
        p25d = p5d + [d for i, d in enumerate(data) if i in p25]
        print('prop: 0.25, num: ', len(p25d))
        write_lines(p25d, f"{DIR}{name}-0.25.txt")
        if name == 'gold':
            print('prop: 1.0, num: ', len(data))
            p100d = data
        else:
            one = set(i for i, lines in enumerate(data) if filter(lines, 1, 2))
            one = one.difference(two).difference(good)
            print('one: ', len(one))
            ids = set(i for i, lines in enumerate(data) if filter(lines, 0, 4))
            ids = ids.difference(one).difference(two).difference(good)
            need = 5985 - len(p25d) - len(left) - len(one)
            print('need: ', need, ', ids: ', len(ids))
            p100 = left.union(set(random.sample(ids, need))).union(one)
            p100d = p25d + [d for i, d in enumerate(data) if i in p100]
            print('prop: 1.0, num: ', len(p100d))
        write_lines(p100d, f"{DIR}{name}-1.0.txt")
        return

    multi(gold, 'gold')
    print('\n')
    multi(conll_rest, 'rest')

    # for p in (0.01, 0.05, 0.25, 1.0):  60, 300, 1497, 5985
    #     num = int(ALL * p + 1)
    #     print('prop: ', p, ', num: ', num)
    #     sampled = random.sample(conll_rest, num)
    #     write_lines(sampled, f"{DIR}rest-{p}.txt")
    #     sampled = random.sample(gold, num)
    #     write_lines(sampled, f"{DIR}gold-{p}.txt")
    return


def main():
    # sample_data()
    return


if __name__ == "__main__":
    main()
