from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

from enum import Enum
from typing import Any, Iterable

import torch.nn.functional as F
from torch import Tensor, nn

from sentence_transformers.SentenceTransformer import SentenceTransformer
from myimports import *
from sentence_transformers.losses.TripletLoss import TripletDistanceMetric

class CircleLoss(nn.Module):
    def __init__(
        self, model: SentenceTransformer, distance_metric=TripletDistanceMetric.COSINE, scale:float=32, margin:float=0.25
    ) -> None:
        # from 
        """
        This class implements Circle loss. Given a triplet of (anchor, positive, negative),
        the loss minimizes the distance between anchor and positive while it maximizes the distance
        between anchor and negative. See https://github.com/qianjinhao/circle-loss/blob/master/circle_loss.py

        scale and margin are important hyperparameter and need to be tuned respectively.

        Args:
            model: SentenceTransformerModel
            distance_metric: Function to compute distance between two
                embeddings. The class TripletDistanceMetric contains
                common distance metrices that can be used.
            scale
            margin

        References:
            - For further details, see: 'Circle Loss: A Unified Perspective of Pair Similarity Optimization'

        Requirements:
            1. (anchor, positive, negative) triplets

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | (anchor, positive, negative) triplets | none   |
            +---------------------------------------+--------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                    "negative": ["It's quite rainy, sadly.", "She walked to the store."],
                })
                loss = losses.CircleLoss(model=model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.distance_metric = distance_metric
        self.scale = scale
        self.margin=margin

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
     
        # m = labels.size(0)
        # mask = labels.expand(m, m).t().eq(labels.expand(m, m)).float()
        # pos_mask = mask.triu(diagonal=1)
        # neg_mask = (mask - 1).abs_().triu(diagonal=1)
        # if self.distance_metric == TripletDistanceMetric.EUCLIDEAN:
        #     sim_mat = torch.matmul(sentence_features, torch.t(sentence_features))
        # elif self.distance_metric == TripletDistanceMetric.COSINE:
        #     sentence_features = F.normalize(sentence_features)
        #     sim_mat = sentence_features.mm(sentence_features.t())
        # else:
        #     raise ValueError('This similarity is not implemented.')

        # pos_pair_ = sim_mat[pos_mask == 1]
        # neg_pair_ = sim_mat[neg_mask == 1]

        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        rep_anchor, rep_pos, rep_neg = reps

        if(self.distance_metric == TripletDistanceMetric.COSINE):
            rep_anchor = F.normalize(rep_anchor, p=2, dim=1)
            rep_pos = F.normalize(rep_pos, p=2, dim=1)
            rep_neg = F.normalize(rep_neg, p=2, dim=1)
            
        pos_pair_ = self.distance_metric(rep_anchor, rep_pos)
        neg_pair_ = self.distance_metric(rep_anchor, rep_neg)

        alpha_p = torch.relu(-pos_pair_ + 1 + self.margin)
        alpha_n = torch.relu(neg_pair_ + self.margin)
        margin_p = 1 - self.margin
        margin_n = self.margin
        loss_p = torch.sum(torch.exp(-self.scale * alpha_p * (pos_pair_ - margin_p)))
        loss_n = torch.sum(torch.exp(self.scale * alpha_n * (neg_pair_ - margin_n)))
        loss = torch.log(1 + loss_p * loss_n)
        return loss
 
    def get_config_dict(self) -> dict[str, Any]:
        distance_metric_name = self.distance_metric.__name__
        for name, value in vars(TripletDistanceMetric).items():
            if value == self.distance_metric:
                distance_metric_name = f"TripletDistanceMetric.{name}"
                break

        return {"distance_metric": distance_metric_name, "scale": self.scale, "margin": self.margin}

    @property
    def citation(self) -> str:
        return """
        @misc{hermans2017defense,
            title={In Defense of the Triplet Loss for Person Re-Identification},
            author={Alexander Hermans and Lucas Beyer and Bastian Leibe},
            year={2017},
            eprint={1703.07737},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
        }
        """