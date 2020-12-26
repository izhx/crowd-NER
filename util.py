
from typing import List, Tuple

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
