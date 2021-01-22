"""
"""

import os
import argparse

_ARG_PARSER = argparse.ArgumentParser(description="我的实验，需要指定配置文件")
_ARG_PARSER.add_argument('--yaml', '-y', type=str, default='cc-pg',
                         help='configuration file path.')
_ARG_PARSER.add_argument('--cuda', '-c', type=str, default='3', help='gpu ids, like: 1,2,3')
_ARG_PARSER.add_argument('--test', '-t', type=bool, default=False, help='只进行测试')
_ARG_PARSER.add_argument('--out', '-o', type=bool, default=False, help='预测结果输出')
_ARG_PARSER.add_argument('--name', '-n', type=str, default=None, help='save name.')
_ARG_PARSER.add_argument('--seed', '-s', type=int, default=123, help='random seed')
_ARG_PARSER.add_argument('--all', '-a', type=bool, default=False, help='all seed ?')
_ARG_PARSER.add_argument('--debug', '-d', default=False, action="store_true")

_ARG_PARSER.add_argument('--adapter_size', type=int, default=None)
_ARG_PARSER.add_argument('--lstm_size', type=int, default=None)
_ARG_PARSER.add_argument('--worker_dim', type=int, default=None)
_ARG_PARSER.add_argument('--pgn_layers', type=int, default=None)
_ARG_PARSER.add_argument('--share_param', type=bool, default=None)

_ARGS = _ARG_PARSER.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = _ARGS.cuda

if _ARGS:
    import random

    import numpy as np
    import torch

    from transformers import BertTokenizer

    import nmnlp
    from nmnlp.common.config import load_yaml
    from nmnlp.common.util import output, cache_path, load_cache, dump_cache
    from nmnlp.common.writer import Writer
    from nmnlp.core import Trainer, Vocabulary
    from nmnlp.core.optim import build_optimizer

    from util import allowed_transition
    from models import build_model
    from datasets import CoNLL03Crowd
else:
    raise Exception('Argument error.')

SEEDS = (123, 456, 789, 686, 666, 233, 1024, 2080, 3080, 3090)

nmnlp.core.trainer.EARLY_STOP_THRESHOLD = 5


def set_seed(seed: int = 123):
    output(f"Process id: {os.getpid()}, cuda: {_ARGS.cuda}, set seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)  # CPU占用过高，但训练速度没快，还没找到问题所在


def run_once(cfg, dataset, vocab, device, writer=None, seed=123):
    model = build_model(vocab=vocab, **cfg.model)
    setattr(model, 'seed', seed)
    para_num = sum([p.numel() for p in model.parameters()])
    output(f'param num: {para_num}, {para_num / 1000000:4f}M')
    model.to(device=device)

    optimizer = build_optimizer(model, **cfg.optim)
    trainer = Trainer(vars(cfg), dataset, vocab, model, optimizer, None, None,
                      writer, device, **cfg.trainer)

    if not _ARGS.test:
        # 训练过程
        trainer.train()
        output(model.metric.data_info)

    trainer.load()
    test_metric = trainer.test(dataset.test)
    return model.metric.best, test_metric


def main():
    cfg = argparse.Namespace(**load_yaml(f"./dev/config/{_ARGS.yaml}.yml"))

    device = torch.device("cuda:0")
    data_kwargs, vocab_kwargs = dict(cfg.data), dict(cfg.vocab)
    use_bert = 'bert' in cfg.model['word_embedding']['name_or_path']

    # 如果用了BERT，要加载tokenizer
    if use_bert:
        tokenizer = BertTokenizer.from_pretrained(
            cfg.model['word_embedding']['name_or_path'],
            do_lower_case=False)
        print("I'm batman!  ",
              tokenizer.tokenize("I'm batman!"))  # [CLS] [SEP]
        data_kwargs['tokenizer'] = tokenizer
        vocab_kwargs['oov_token'] = tokenizer.unk_token
        vocab_kwargs['padding_token'] = tokenizer.pad_token
    else:
        tokenizer = None

    cache_name = _ARGS.yaml
    if not os.path.exists(cache_path(cache_name)):
        dataset = argparse.Namespace(
            **CoNLL03Crowd.build(**cfg.data, tokenizer=tokenizer))
        vocab = Vocabulary.from_data(dataset, **vocab_kwargs)
        vocab.set_field(['O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC'], 'tags')

        if use_bert:
            # 若用BERT，则把words词表替换为BERT的
            vocab.token_to_index['words'] = tokenizer.vocab
            vocab.index_to_token['words'] = tokenizer.ids_to_tokens
        dump_cache((dataset, vocab), cache_name)
    else:
        dataset, vocab = load_cache(cache_name)

    dataset.train.index_with(vocab)
    dataset.dev.index_with(vocab)
    dataset.test.index_with(vocab)

    cfg.model['allowed'] = allowed_transition(vocab)
    cfg.model['output_prediction'] = _ARGS.out

    prefix = _ARGS.name if _ARGS.name else _ARGS.yaml
    info = list()
    if _ARGS.debug:
        log_dir = None
        cfg.trainer['save_strategy'] = 'no'
    else:
        # log_dir = f"./dev/tblog/{prefix}"
        # if not os.path.exists(log_dir):
        #     os.mkdir(log_dir)
        log_dir = None

    for k in ('lstm_size', 'adapter_size', 'pgn_layers', 'worker_dim'):
        p = getattr(_ARGS, k)
        if p is not None:
            cfg.model[k] = p
            prefix += f'-l{p}'

    if _ARGS.share_param is not None:
        cfg.model['share_param'] = _ARGS.share_param
        prefix += 'share'

    seeds = SEEDS if _ARGS.all else [_ARGS.seed]
    for seed in seeds:
        print('\n')
        set_seed(seed)
        cfg.trainer['prefix'] = f"{prefix}_{seed}"
        if 'pre_train_path' not in cfg.trainer:
            cfg.trainer['pre_train_path'] = os.path.normpath(
                f"./dev/model/{cfg.trainer['prefix']}_best.pth")
        writer = Writer(log_dir, str(seed), 'tensorboard') if log_dir else None
        info.append(run_once(cfg, dataset, vocab, device, writer, seed))

    # print('\nAVG DEV: ', merge_dicts([i[0] for i in info], avg=True))
    # print('AVG TEST: ', merge_dicts([i[1] for i in info], avg=True))


if __name__ == "__main__":
    main()
