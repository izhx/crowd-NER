import os
import shutil
from typing import Dict, Tuple, List, Any, cast, Union

import torch
from torch.nn import Module, Parameter, ParameterList, Linear, Embedding, init
from torch.nn.modules.loss import CrossEntropyLoss

from nmnlp.core import Vocabulary
from nmnlp.core.trainer import format_metric, output
from nmnlp.embedding import build_word_embedding
from nmnlp.modules.linear import NonLinear
from nmnlp.modules.adapter import AdapterBertModel
from nmnlp.modules.dropout import WordDropout
from nmnlp.modules.encoder import LstmEncoder

from grl import GradientReverseLayer
from metric import ExactMatch
from conditional_random_field import ConditionalRandomField


def build_model(name, **kwargs):
    m = {
        'ad': AdapterModel,
        'pg': PGNModel,
        'adv': AdvModel,
    }
    return m[name](**kwargs)


def tensor_like(data, t: torch.Tensor) -> torch.Tensor:
    tensor = torch.zeros_like(t)
    for i, l in enumerate(data):
        tensor[i, :len(l)] = torch.tensor(l, dtype=t.dtype, device=t.device)
    return tensor


class CRF(Module):
    """ CRF classifier."""
    def __init__(
        self,
        num_tags: int,
        input_dim: int = 0,
        top_k: int = 1,
        reduction='mean',
        constraints: List[Tuple[int, int]] = None,
        include_start_end_transitions: bool = True
    ) -> None:
        super().__init__()
        if input_dim > 0:
            self.tag_projection = Linear(input_dim, num_tags)
        else:
            self.tag_projection = None

        self.base = ConditionalRandomField(
            num_tags, reduction, constraints, include_start_end_transitions)
        self.top_k = top_k

    def forward(
        self, inputs: torch.FloatTensor, mask: torch.LongTensor,
        labels: torch.LongTensor = None, reduction: str = None,
    ) -> Dict[str, Any]:
        bool_mask = mask.bool()

        if self.tag_projection:
            inputs = self.tag_projection(inputs)
        scores = inputs * mask.unsqueeze(-1)

        if labels is None:
            loss = torch.tensor(0, dtype=torch.float, device=inputs.device)
        else:
            # Add negative log-likelihood as loss
            loss = self.base(scores, labels, bool_mask, reduction)

        best_paths = self.base.viterbi_tags(scores, bool_mask, top_k=self.top_k)
        # Just get the top tags and ignore the scores.
        tags = cast(List[List[int]], [x[0][0] for x in best_paths])

        return dict(scores=scores, predicted_tags=tags, loss=loss)


class Tagger(Module):
    """
    a
    """
    def __init__(
        self,
        vocab: Vocabulary,
        word_embedding: Dict[str, Any],
        transform_dim: int = 0,
        lstm_size: int = 400,
        input_namespace: str = 'words',
        label_namespace: str = 'tags',
        save_embedding: bool = False,
        allowed: List[Tuple[int, int]] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.word_embedding = build_word_embedding(
            num_embeddings=vocab.size_of(input_namespace), vocab=vocab, **word_embedding)
        feat_dim: int = self.word_embedding.output_dim

        if transform_dim > 0:
            self.word_transform = NonLinear(feat_dim, transform_dim)
            feat_dim: int = transform_dim
        else:
            self.word_transform = None

        self.lstm = LstmEncoder(feat_dim, lstm_size, num_layers=1)

        num_tags = vocab.size_of(label_namespace)
        self.tag_projection = Linear(self.lstm.output_dim, num_tags)

        self.word_dropout = WordDropout(0.20)
        self.crf = CRF(num_tags, constraints=allowed)
        self.metric = ExactMatch(
            vocab.index_of('O', label_namespace),
            vocab.token_to_index[label_namespace],
            True, True)
        self.save_embedding = save_embedding
        self.id_to_label = vocab.index_to_token[label_namespace]
        self.epoch = 0

    def forward(
        self, words: torch.Tensor, lengths: torch.Tensor,
        mask: torch.Tensor = None, tags: torch.Tensor = None, **kwargs
    ) -> Dict[str, Any]:
        embedding = self.word_embedding(words, mask=mask, **kwargs)
        if self.word_transform is not None:
            embedding = self.word_transform(embedding)
        embedding = self.word_dropout(embedding)
        feature = self.lstm(embedding, lengths, **kwargs)
        scores = self.tag_projection(feature)
        output_dict = self.crf(scores, mask, tags)

        if tags is not None and tags.dim() == 2:
            output_dict = self.add_metric(output_dict, tags, lengths)

        return output_dict

    def add_metric(self, output_dict, tags, lengths, prefix=''):
        prediction = tensor_like(output_dict['predicted_tags'], tags)
        output_dict['metric'] = getattr(self, prefix + "metric")(prediction, tags, lengths)
        return output_dict

    def before_train_once(self, kwargs):
        self.epoch = kwargs['epoch']

    def after_epoch_end(self, kwargs):
        writer, metric = kwargs['self'].writer, kwargs['metric']
        output(format_metric(metric))
        if writer:
            writer.add_scalars('Metric_Train', metric, kwargs['epoch'])

    def save(self, path):
        state_dict = self.state_dict()
        if not self.save_embedding:
            state_dict = {k: v for k, v in state_dict.items() if not self.drop_param(k)}
        torch.save(state_dict, path)

    def load(self, path_or_state, device):
        if isinstance(path_or_state, str):
            path_or_state = torch.load(path_or_state, map_location=device)
        info = self.load_state_dict(path_or_state, strict=False)
        missd = [i for i in info[0] if not self.drop_param(i)]
        if missd:
            print(missd)
        # print("model loaded.")

    def drop_param(_, name: str):
        return name.startswith('word_embedding.bert')


class AdapterModel(Tagger):
    def __init__(self,
                 vocab: Vocabulary,
                 adapter_size: int = 128,
                 external_param: Union[bool, List[bool]] = False,
                 output_prediction: bool = False,
                 **kwargs):
        super().__init__(vocab, **kwargs)
        self.word_embedding = AdapterBertModel(
            self.word_embedding.bert, adapter_size, external_param)
        self.output_prediction = output_prediction

    def drop_param(_, name: str):
        return super().drop_param(name) and 'LayerNorm' not in name

    def before_time_start(self, _, trainer, kwargs):
        if self.output_prediction:
            out_dir = f"dev/out/{trainer.prefix}/"
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            os.mkdir(out_dir)
            setattr(self, 'out_dir', out_dir)

    def before_next_batch(self, kwargs):
        if self.training or not self.output_prediction:
            return
        epoch, batch, out = kwargs['epoch'], kwargs['batch'], kwargs['output_dict']
        path = f"{self.out_dir}/{'test' if epoch is None else 'dev'}-{self.epoch}.txt"
        with open(path, mode='a') as file:
            for ins, pred in zip(batch, out['predicted_tags']):
                for w, li, pi, in zip(ins['text'], ins['tags'], pred):
                    file.write(f"{w}\t{self.id_to_label[li]}\t{self.id_to_label[pi]}\n")
                file.write('\n')
        return


class PGNModel(AdapterModel):
    def __init__(self,
                 vocab: Vocabulary,
                 annotator_dim: int = 8,
                 annotator_num: int = 47,
                 adapter_size: int = 128,
                 num_adapters: int = 12,
                 batched_param: bool = False,
                 share_param: bool = False,
                 same_embedding: bool = False,
                 **kwargs):
        super().__init__(vocab, adapter_size, [True] * num_adapters, **kwargs)
        if same_embedding:
            w = torch.randn(annotator_dim).expand(annotator_num, -1).contiguous()
        else:
            w = torch.randn(annotator_num, annotator_dim)
        self.annotator_embedding = Embedding(
            annotator_num, annotator_dim, _weight=w)  # max_norm=1.0

        dim = self.word_embedding.output_dim
        size = [2] if share_param else [num_adapters, 2]
        self.weight = ParameterList([
            Parameter(torch.Tensor(*size, adapter_size, dim, annotator_dim)),
            Parameter(torch.zeros(*size, adapter_size, annotator_dim)),
            Parameter(torch.Tensor(*size, dim, adapter_size, annotator_dim)),
            Parameter(torch.zeros(*size, dim, annotator_dim)),
        ])
        self.reset_parameters()
        self.adapter_size = adapter_size
        self.num_adapters = num_adapters
        self.batched_param = batched_param
        self.share_param = share_param

    def reset_parameters(self):
        # bound = 1e-2
        # init.uniform_(self.weight[0], -bound, bound)
        # init.uniform_(self.weight[2], -bound, bound)
        init.normal_(self.weight[0], std=1e-3)
        init.normal_(self.weight[2], std=1e-3)

    def set_annotator(self, user: torch.LongTensor):
        if self.training and user is not None and user[0].item() != -1:  # expert = -1
            ann_emb = self.annotator_embedding(user[0] if self.batched_param else user)
        elif hasattr(self, 'scores'):
            weight = self.scores.softmax(0).unsqueeze(-1)
            ann_emb = self.annotator_embedding.weight.mul(weight).sum(0)
        elif hasattr(self, 'ann_id'):
            ann_emb = self.annotator_embedding.weight[self.ann_id]
        else:
            ann_emb = self.annotator_embedding.weight.mean(0)
        self.set_adapter_parameter(ann_emb)

    def set_adapter_parameter(self, embedding: torch.Tensor):
        def batch_matmul(w: torch.Tensor, e):
            ALPHA = "ijklmnopqrstuvwxyz"
            dims = ALPHA[:w.dim() - 1]
            i = 1 if self.share_param else 2
            return torch.einsum(f"{dims}a,ba->{dims[:i] + 'b' + dims[i:]}", w, e)

        matmul = batch_matmul if embedding.dim() == 2 else torch.matmul
        embedding = embedding.softmax(-1)
        param_list = [matmul(w, embedding) for w in self.weight]

        for i, adapters in enumerate(self.word_embedding.adapters[-self.num_adapters:]):
            for j, adapter in enumerate(adapters):
                params: List[torch.Tensor] = [p[j] if self.share_param else p[i, j] for p in param_list]
                setattr(adapter, 'params', params)

    def forward(
        self, words: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor,
        aid: torch.LongTensor = None, embedding: torch.Tensor = None,
        tags: torch.Tensor = None, **kwargs
    ) -> Dict[str, Any]:
        if embedding is None:
            self.set_annotator(aid)
        else:
            self.set_adapter_parameter(embedding)
        output_dict = super().forward(words, lengths, mask, tags, **kwargs)
        if self.training and tags.dim() == 2:
            output_dict = self.add_metric(output_dict, tags, lengths)
        return output_dict


class AdvModel(PGNModel):
    def __init__(self,
                 vocab: Vocabulary,
                 lstm_size: int = 400,
                 annotator_num: int = 47,
                 **kwargs):
        super().__init__(vocab, lstm_size=lstm_size, **kwargs)
        self.adv_lstm = LstmEncoder(self.word_embedding.output_dim, lstm_size, num_layers=1)
        dim = self.adv_lstm.output_dim
        self.feature_projection = Linear(dim * 2, dim, bias=True)
        self.annotator_projection = Linear(dim, annotator_num)
        self.ce = CrossEntropyLoss()
        self.grl = GradientReverseLayer()

    def forward(
        self, words: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor,
        aid: torch.LongTensor = None, embedding: torch.Tensor = None,
        tags: torch.Tensor = None, **kwargs
    ) -> Dict[str, Any]:
        if embedding is None:
            self.set_annotator(aid)
        else:
            self.set_adapter_parameter(embedding)

        # return super().forward(words, lengths, mask, tags, **kwargs)

        feature = self.word_embedding(words, mask=mask, **kwargs)
        feature = self.word_dropout(feature)
        common_feature = self.lstm(feature, lengths, **kwargs)
        special_feature = self.adv_lstm(feature, lengths, **kwargs)
        all_feature = torch.cat((special_feature, common_feature), dim=-1)
        all_feature = self.feature_projection(all_feature)
        scores = self.tag_projection(all_feature)
        output_dict: Dict = self.crf(scores, mask, tags)

        if aid is not None:
            sentence_feature = self.grl(common_feature[:, 0])
            ann_scores = self.annotator_projection(sentence_feature)
            loss_2 = self.ce(ann_scores, aid)
            ann_scores = self.annotator_projection(special_feature[:, 0])
            loss_3 = self.ce(ann_scores, aid)
            loss_1 = output_dict.pop('loss')
            loss = loss_1 + loss_2 + loss_3
            output_dict.update(loss=loss)

        if tags is not None and tags.dim() == 2:
            output_dict = self.add_metric(output_dict, tags, lengths)
        return output_dict
