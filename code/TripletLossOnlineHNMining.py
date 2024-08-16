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
    HARDNEGATIVE = 1        #only use triplets in batch where pos_pair<neg_pair (harder problem, use fewer rows per batch
    SEMIHARDNEGATIVE = 2    #only use triplets in batch where pos_pair<neg_pair < pos_pair+margin (easier problem)
    BOTH =3                 #same as triplet loss in batch

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

        ### pytorch-metric-learning stuff ###
        distance = distances.CosineSimilarity()
        reducer = reducers.ThresholdReducer(low=0)
        loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
        mining_func = miners.TripletMarginMiner(
        margin=0.2, distance=distance, type_of_triplets="semihard"
        )

        #from https://quaterion.qdrant.tech/tutorials/triplet_loss_trick
        #prevents mode collapse where model maps all inputs to same point
        losses = F.relu((distance_pos - distance_neg)/distance_neg.mean() + self.triplet_margin)

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
            #for semi hard negative mining, only use triplets in batch where distance(pos_pair)<distance(neg_pair) < distance(pos_pair)+margin
            mask = ((distance_pos<distance_neg) & (distance_neg<(distance_pos+self.triplet_margin)))
       
        elif self.online_mining_type == OnlineMineingType.HARDNEGATIVE:
            #for hard negatives can either choose just negatives where distance_neg is smallest (how many of these though?)
            #OR choose all negatives where distance_neg<distance_pos, meaning the negative is closest to anchor than the positive
            mask = (distance_neg<distance_pos)
        else:
            #its BOTH
            #do both semi and hard negatives
            mask = torch.ones_like(distance_pos, dtype=torch.bool)

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
@misc{Extended from Hugging Face TripletLoss
}
"""
