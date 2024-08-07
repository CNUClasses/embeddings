from __future__ import annotations

from enum import Enum
from typing import Any, Iterable

import torch.nn.functional as F
from torch import Tensor, nn
import torch

from sentence_transformers.SentenceTransformer import SentenceTransformer
from enum import Enum

class TripletDistanceMetric(Enum):
    """The metric for the triplet loss"""

    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)


class OnlineMineingType(Enum):
    HARDNEGATIVE = 1    #only use triplets in batch where pos_pair<neg_pair (harder problem, use fewer rows per batch
    SEMIHARDNEGATIVE = 2 #only use triplets in batch where pos_pair<neg_pair < margin (easier problem)

class TripletLossOnlineHNMining(nn.Module):
    def __init__(
        self, model: SentenceTransformer, distance_metric=TripletDistanceMetric.COSINE, triplet_margin: float = 0.2, OnLineMiningType=OnlineMineingType.HARDNEGATIVE, verbose=True
    ) -> None:
        """
        This class implements triplet loss with semi Hard, and Hard online mining. Given a triplet of (anchor, positive, negative),
        the loss minimizes the distance between anchor and positive while it maximizes the distance
        between anchor and negative. It compute the following loss function:

        ``loss = max(||anchor - positive|| - ||anchor - negative|| + margin, 0)``.

        Margin is an important hyperparameter and needs to be tuned respectively.

        Args:
            model: SentenceTransformerModel
            distance_metric: Function to compute distance between two
                embeddings. The class TripletDistanceMetric contains
                common distance metrices that can be used.
            triplet_margin: The negative should be at least this much
                further away from the anchor than the positive.

        References:
            - For further details, see: https://en.wikipedia.org/wiki/Triplet_loss

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
                loss = losses.TripletLoss(model=model)

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
        self.triplet_margin = triplet_margin
        self.online_mining_type = OnLineMiningType
        self.verbose=verbose
        self.stablecount=0
        self.decreaseafternumbatches=10 

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

        rep_anchor, rep_pos, rep_neg = reps
        distance_pos = self.distance_metric(rep_anchor, rep_pos)
        distance_neg = self.distance_metric(rep_anchor, rep_neg)

        losses = F.relu(distance_pos - distance_neg + self.triplet_margin)

        # if self.online_mining_type == OnlineMineingType.SEMIHARDNEGATIVE:
        #     #for semi hard negative mining, only use triplets in batch where pos_pair<neg_pair < margin
        #     mask = (distance_pos<distance_neg) & (distance_neg<self.triplet_margin)
        #     len_losses_upper=int(len(losses)*.25)
          
        #     num_losses_considered=torch.sum(mask).item()
        #     if(num_losses_considered>=len_losses_upper):
        #         #if we are getting too many losses lets tighten the margin by 10% for next time
        #         self.triplet_margin=self.triplet_margin-(self.triplet_margin*.1)
        #         # print(f"   decreased margin to {self.triplet_margin}")
        #     elif(num_losses_considered==0):
        #         #if we are getting too few losses lets loosen the margin by 10% for next time
        #         self.triplet_margin=self.triplet_margin+(self.triplet_margin*.1)
        #         # print(f"     increased margin to {self.triplet_margin}")
        #     else:
        #         #if we are getting the right number of losses, lets keep the margin the same
        #         #after a while decrease triplet_margin
        #         self.stablecount+=1
        #         if(self.stablecount >= self.decreaseafternumbatches):
        #             self.stablecount=0
        #             self.triplet_margin=self.triplet_margin-(self.triplet_margin*.01)
        #         # print(f"Stable margin {self.triplet_margin}, used {num_losses_considered} losses")
        #         pass

        if self.online_mining_type == OnlineMineingType.SEMIHARDNEGATIVE:
            #for semi hard negative mining, only use triplets in batch where pos_pair<neg_pair < margin
            mask = (distance_pos<distance_neg) & (distance_neg<self.triplet_margin)
       
        if self.online_mining_type == OnlineMineingType.HARDNEGATIVE:
            #for hard negatives can either choose just negatives where distance_neg is smallest (how many of these though?)
            #OR choose all negatives where distance_neg<distance_pos, meaning the negative is closest to anchor than the positive
            mask = (distance_neg<distance_pos)

        losses = losses * mask.float()
        mean_losses=losses.mean()
       
        #DUPLICATE LINE REFACTOR
        num_losses_considered=torch.sum(mask).item()
        if(num_losses_considered>0):
            mean_losses=mean_losses*len(losses)/num_losses_considered
        
        return mean_losses
        # return mn

    def get_config_dict(self) -> dict[str, Any]:
        distance_metric_name = self.distance_metric.__name__
        for name, value in vars(TripletDistanceMetric).items():
            if value == self.distance_metric:
                distance_metric_name = f"TripletDistanceMetric.{name}"
                break

        return {"distance_metric": distance_metric_name, "triplet_margin": self.triplet_margin}

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


# from __future__ import annotations
# import torch
# from torch import nn
# import torch.nn.functional as F

# from enum import Enum
# from typing import Any, Iterable

# import torch.nn.functional as F
# from torch import Tensor, nn

# from sentence_transformers.SentenceTransformer import SentenceTransformer
# from myimports import *
# from sentence_transformers.losses.TripletLoss import TripletDistanceMetric

# class CircleLoss(nn.Module):
#     def __init__(
#         self, model: SentenceTransformer, distance_metric=TripletDistanceMetric.COSINE, scale:float=32, margin:float=0.25
#     ) -> None:
#         # from 
#         """
#         This class implements Circle loss. Given a triplet of (anchor, positive, negative),
#         the loss minimizes the distance between anchor and positive while it maximizes the distance
#         between anchor and negative. See https://github.com/qianjinhao/circle-loss/blob/master/circle_loss.py

#         scale and margin are important hyperparameter and need to be tuned respectively.

#         Args:
#             model: SentenceTransformerModel
#             distance_metric: Function to compute distance between two
#                 embeddings. The class TripletDistanceMetric contains
#                 common distance metrices that can be used.
#             scale
#             margin

#         References:
#             - For further details, see: 'Circle Loss: A Unified Perspective of Pair Similarity Optimization'

#         Requirements:
#             1. (anchor, positive, negative) triplets

#         Inputs:
#             +---------------------------------------+--------+
#             | Texts                                 | Labels |
#             +=======================================+========+
#             | (anchor, positive, negative) triplets | none   |
#             +---------------------------------------+--------+

#         Example:
#             ::

#                 from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
#                 from datasets import Dataset

#                 model = SentenceTransformer("microsoft/mpnet-base")
#                 train_dataset = Dataset.from_dict({
#                     "anchor": ["It's nice weather outside today.", "He drove to work."],
#                     "positive": ["It's so sunny.", "He took the car to the office."],
#                     "negative": ["It's quite rainy, sadly.", "She walked to the store."],
#                 })
#                 loss = losses.CircleLoss(model=model)

#                 trainer = SentenceTransformerTrainer(
#                     model=model,
#                     train_dataset=train_dataset,
#                     loss=loss,
#                 )
#                 trainer.train()
#         """
#         super().__init__()
#         self.model = model
#         self.distance_metric = distance_metric
#         self.scale = scale
#         self.margin=margin

#     def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
     
#         # m = labels.size(0)
#         # mask = labels.expand(m, m).t().eq(labels.expand(m, m)).float()
#         # pos_mask = mask.triu(diagonal=1)
#         # neg_mask = (mask - 1).abs_().triu(diagonal=1)
#         # if self.distance_metric == TripletDistanceMetric.EUCLIDEAN:
#         #     sim_mat = torch.matmul(sentence_features, torch.t(sentence_features))
#         # elif self.distance_metric == TripletDistanceMetric.COSINE:
#         #     sentence_features = F.normalize(sentence_features)
#         #     sim_mat = sentence_features.mm(sentence_features.t())
#         # else:
#         #     raise ValueError('This similarity is not implemented.')

#         # pos_pair_ = sim_mat[pos_mask == 1]
#         # neg_pair_ = sim_mat[neg_mask == 1]

#         reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
#         rep_anchor, rep_pos, rep_neg = reps

#         if(self.distance_metric == TripletDistanceMetric.COSINE):
#             rep_anchor = F.normalize(rep_anchor, p=2, dim=1)
#             rep_pos = F.normalize(rep_pos, p=2, dim=1)
#             rep_neg = F.normalize(rep_neg, p=2, dim=1)
            
#         pos_pair_ = self.distance_metric(rep_anchor, rep_pos)
#         neg_pair_ = self.distance_metric(rep_anchor, rep_neg)

#         #for semi hard negative mining, only use triplets in batch where pos_pair<neg_pair < margin
#         mask= (pos_pair_<neg_pair_) & (neg_pair_<self.margin)
#         dis= pos_pair_-neg_pair_
#         mask=dis<self.margin


#         #for hard negative mining, only use triplets in batch where neg_pair<pos_pair

#         alpha_p = torch.relu(-pos_pair_ + 1 + self.margin)
#         alpha_n = torch.relu(neg_pair_ + self.margin)
#         margin_p = 1 - self.margin
#         margin_n = self.margin
#         loss_p = torch.sum(torch.exp(-self.scale * alpha_p * (pos_pair_ - margin_p)))
#         loss_n = torch.sum(torch.exp(self.scale * alpha_n * (neg_pair_ - margin_n)))
#         loss = torch.log(1 + loss_p * loss_n)
#         return loss
 
#     def get_config_dict(self) -> dict[str, Any]:
#         distance_metric_name = self.distance_metric.__name__
#         for name, value in vars(TripletDistanceMetric).items():
#             if value == self.distance_metric:
#                 distance_metric_name = f"TripletDistanceMetric.{name}"
#                 break

#         return {"distance_metric": distance_metric_name, "scale": self.scale, "margin": self.margin}

#     @property
#     def citation(self) -> str:
#         return """
#         @misc{hermans2017defense,
#             title={In Defense of the Triplet Loss for Person Re-Identification},
#             author={Alexander Hermans and Lucas Beyer and Bastian Leibe},
#             year={2017},
#             eprint={1703.07737},
#             archivePrefix={arXiv},
#             primaryClass={cs.CV}
#         }
#         """