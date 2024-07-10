#see https://www.pinecone.io/learn/series/nlp/fine-tune-sentence-transformers-mnr/

import utils
import torch
import datasets
import logging

from datetime import datetime
startTime = datetime.now()

#setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'tmp.log', encoding='utf-8', level=logging.DEBUG)

from datasets import load_dataset, concatenate_datasets
test_dataset = load_dataset("json", data_files="tst.json", split="train")
train_dataset = load_dataset("json", data_files="trn.json", split="train")
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

from sentence_transformers import InputExample
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
bert = models.Transformer(f'{utils.modelname}')
pooler = models.Pooling(
    bert.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)
model = SentenceTransformer(modules=[bert, pooler],
                            device="cuda:0" if torch.cuda.is_available() else "cpu",)

#create loss
from sentence_transformers import losses
loss = losses.MultipleNegativesRankingLoss(model)

#evaluate base model
from sentence_transformers.evaluation import InformationRetrievalEvaluator
dev_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name=f'{utils.modelname}',
)
logger.info(f'Base {utils.modelname} performance:')
tmptme=datetime.now()
logger.info(dev_evaluator(model))
logger.info(f'Total time for base model eval={datetime.now() - tmptme}')

#train model
epochs = 4
warmup_steps = int(len(loader) * epochs * 0.1)

model.fit(
    train_objectives=[(loader, loss)],
    epochs=epochs,
    warmup_steps=warmup_steps,
    output_path=f'./{utils.modelname}',
    show_progress_bar=True
) 

logger.info(f'Finetuned {utils.modelname} performance:')
logger.info(dev_evaluator(model))
logger.info(f'Total time for script={datetime.now() - startTime}')
