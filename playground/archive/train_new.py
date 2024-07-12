#see https://www.pinecone.io/learn/series/nlp/fine-tune-sentence-transformers-mnr/

import utils as ut
import torch
import datasets
import logging
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import InputExample


from datetime import datetime
startTime = datetime.now()


#setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename=f"LOG_{ut.modelname.split('/')[-1]}.log", encoding='utf-8', level=logging.DEBUG, filemode="w",)

from datasets import load_dataset, concatenate_datasets
test_dataset = load_dataset("json", data_files="../data/tst.json", split="train")
train_dataset = load_dataset("json", data_files="../data/trn.json", split="train")
corpus_dataset = concatenate_datasets([train_dataset, test_dataset])

# Convert the datasets to dictionaries
corpus = dict(
    zip(corpus_dataset["id"], corpus_dataset["positive"])
)  # Our corpus (cid => document)
queries = dict(
    zip(test_dataset["id"], test_dataset["anchor"])
)  

# Create a mapping of relevant document (1 in our case) for each query
relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
for q_id in queries:
    relevant_docs[q_id] = [q_id]
 
logger.info(f"{len(train_dataset)} rows")
# logger.info(train_dataset)
# logger.info(train_dataset[0])


from tqdm.auto import tqdm  # so we see progress bar
train_samples = []
for row in tqdm(train_dataset):
    train_samples.append(InputExample(
        texts=[row['positive'], row['anchor']]
    ))

#get train dataset
from sentence_transformers import datasets
batch_size = 32
loader = datasets.NoDuplicatesDataLoader(
    train_samples, batch_size=batch_size)

#create model
from sentence_transformers import models, SentenceTransformer
bert = models.Transformer(f'{ut.modelname}')
pooler = models.Pooling(
    bert.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)
model = SentenceTransformer(modules=[bert, pooler],
                            device="cuda:0" if torch.cuda.is_available() else "cpu",)

#create loss
from sentence_transformers import losses
loss = losses.MultipleNegativesRankingLoss(model)



# 5. (Optional) Specify training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=f"models/{ut.modelname.split('/')[-1]}",
    # Optional training parameters:
    num_train_epochs=5,
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
    run_name="f{ut.modelname.split('/')[-1]}",  # Will be used in W&B if `wandb` is installed
)


#evaluate base model
from sentence_transformers.evaluation import InformationRetrievalEvaluator
dev_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name=f'{ut.modelname}',
)
logger.info(f'Base {ut.modelname} performance:')
tmptme=datetime.now()
logger.info(dev_evaluator(model))
logger.info(f'Total time for base model eval={datetime.now() - tmptme}')

#train model
# epochs = 4
# warmup_steps = int(len(loader) * epochs * 0.1)

# model.fit(
#     train_objectives=[(loader, loss)],
#     epochs=epochs,
#     warmup_steps=warmup_steps,
#     output_path=f'./{ut.modelname}',
#     show_progress_bar=True
# ) 

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

logger.info(f'Finetuned {ut.modelname} performance:')
logger.info(dev_evaluator(model))
logger.info(f'Total time for script={datetime.now() - startTime}')

# 8. Save the trained model
model.save_pretrained(f"models/{ut.modelname.split('/')[-1]}")
