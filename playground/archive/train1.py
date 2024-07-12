
import utils
import torch
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)

from datasets import load_dataset, concatenate_datasets
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator

# 1. Load a model to finetune with 2. (Optional) model card data
# model = SentenceTransformer(utils.modelname)
model = SentenceTransformer(f'{utils.modelname}',
                            device="cuda:0" if torch.cuda.is_available() else "cpu",
                            model_kwargs={"attn_implementation": "sdpa"},
                            )

# 3. Load a dataset to finetune on
# load test dataset
test_dataset = load_dataset("json", data_files="tst.json", split="train")
train_dataset = load_dataset("json", data_files="trn.json", split="train")
corpus_dataset = concatenate_datasets([train_dataset, test_dataset])

# Convert the datasets to dictionaries
corpus = dict(
    zip(corpus_dataset["id"], corpus_dataset["positive"])
)  # Our corpus (cid => document)
queries = dict(
    zip(test_dataset["id"], test_dataset["anchor"])
)  # Our queries (qid => question)

# del test_dataset, train_dataset, corpus_dataset
# utils.clean_up()

# Create a mapping of relevant document (1 in our case) for each query
relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
for q_id in queries:
    relevant_docs[q_id] = [q_id]
 
# 3. Load a dataset to finetune on
# dataset = load_dataset("sentence-transformers/all-nli", "triplet")
# train_dataset = dataset["train"].select(range(100_000))
# eval_dataset = dataset["dev"]
# test_dataset = dataset["test"]

# 4. Define a loss function
loss = MultipleNegativesRankingLoss(model)

# 5. (Optional) Specify training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=f"models/{utils.modelname}",
    # Optional training parameters:
    num_train_epochs=4,
    per_device_train_batch_size=32,
     gradient_accumulation_steps=16,             # for a global batch size of 512
    per_device_eval_batch_size=16,
    warmup_ratio=0.1,                           # warmup ratio
    learning_rate=2e-5,
    lr_scheduler_type="cosine",                 # use constant learning rate scheduler
    optim="adamw_torch_fused",                  # use fused adamw optimizer
    fp16=True,                                  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,                                 # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="epoch",                      # evaluate after each epoch
    save_strategy="epoch",                      # save after each epoch
    logging_steps=10,                           # log every 10 steps
    save_total_limit=3,                         # save only the last 3 models
    load_best_model_at_end=True,                # load the best model when training ends
    run_name=f"{utils.modelname}",  # Will be used in W&B if `wandb` is installed
)

# 6. (Optional) Create an evaluator & evaluate the base model
dev_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name=f"utils.modelname",
)
print(dev_evaluator(model))


# 7. Create a trainer & train
# load train dataset again
train_dataset = load_dataset("json", data_files="trn.json", split="train")

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset.select_columns(
        ["positive", "anchor"]
    ),
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

# (Optional) Evaluate the trained model on the test set

dev_evaluator(model)

# 8. Save the trained model
model.save_pretrained("models/mpnet-base-all-nli-triplet/final")

# 9. (Optional) Push it to the Hugging Face Hub
# model.push_to_hub("mpnet-base-all-nli-triplet")