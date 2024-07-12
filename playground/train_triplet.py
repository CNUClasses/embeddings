#see https://www.pinecone.io/learn/series/nlp/fine-tune-sentence-transformers-mnr/

import utils as ut
import torch
import datasets
import logging
import pandas as pd

from datetime import datetime
startTime = datetime.now()

#setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename=f"LOG_{ut.modelname.split('/')[-1]}.log", encoding='utf-8', level=logging.DEBUG, filemode="w",)


from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import TripletLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
from datasets import Dataset

# 1. Load a model to finetune with 2. (Optional) model card data
model = SentenceTransformer(f"./models/{ut.modelname.split('/')[-1]}",device="cuda:0" if torch.cuda.is_available() else "cpu",)

# 3. Load a dataset to finetune on
# train_df=pd.read_json('../data/trn_with_hard_negatives.json')
train_dataset = load_dataset("json", data_files="../data/trn_with_hard_negatives.json", split="train")
eval_dataset = load_dataset("json", data_files="../data/eval_with_hard_negatives.json", split="train")
test_dataset = load_dataset("json", data_files="../data/tst_with_hard_negatives.json", split="train")

# 4. Define a loss function
loss = TripletLoss(model=model)

# 5. (Optional) Specify training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=f"models/{ut.modelname.split('/')[-1]}_triplet",
    # Optional training parameters:
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name=f"{ut.modelname.split('/')[-1]}_triplet",  # Will be used in W&B if `wandb` is installed
)

# 6. (Optional) Create an evaluator & evaluate the base model
dev_evaluator = TripletEvaluator(
    anchors=test_dataset["anchor"],
    positives=test_dataset["positive"],
    negatives=test_dataset["negative"],
    name="all-nli-dev",
)
dev_evaluator(model)

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

# (Optional) Evaluate the trained model on the test set
test_evaluator = TripletEvaluator(
    anchors=test_dataset["anchor"],
    positives=test_dataset["positive"],
    negatives=test_dataset["negative"],
    name="all-nli-test",
)
test_evaluator(model)

# 8. Save the trained model
model.save_pretrained("models/mpnet-base-all-nli-triplet/final")

# 9. (Optional) Push it to the Hugging Face Hub
model.push_to_hub("mpnet-base-all-nli-triplet")

