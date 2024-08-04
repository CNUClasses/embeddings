from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from myimports import *
import utils as ut
ut.login_hf()   #need this because GPU server keeps going down and scripts fail


from sentence_transformers.util import mine_hard_negatives
from minehardnegativesextended_function import mine_hard_negatives_extended

# 3. Load a dataset to finetune on
train_dataset = load_dataset("json", data_files="../data/trn.json", split="train")
eval_dataset = load_dataset("json", data_files="../data/eval.json", split="train")
test_dataset = load_dataset("json", data_files="../data/tst.json", split="train")

def fixcols(ds):
    ds = ds.rename_column("anchor", "query")
    ds = ds.rename_column("positive", "answer")
    ds = ds.remove_columns(["id"])
    return ds

train_dataset=fixcols(train_dataset)
eval_dataset=fixcols(eval_dataset)
test_dataset=fixcols(test_dataset)

#create a corpus spanning all answer columns
corpus_dataset = concatenate_datasets([train_dataset, eval_dataset, test_dataset])
corpus_dataset=corpus_dataset.remove_columns(["query"])

#load a model to calculate similarities with
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_hn_dataset(ds:"Dataset", corpus_dataset:"Dataset"):
    dataset = mine_hard_negatives_extended(
            dataset=ds,
            model=model,
            corpus=corpus_dataset,
            range_min=0,  #was 10
            # range_max=50, #was 50
            # max_score=0.,  #was .8
            margin=0,  #gurantees that the negative is always further away than the positive
            num_negatives=3,
            sampling_strategy="random",
            batch_size=128,
            use_faiss=False,
            verbose=True,
        )

    #add a new column that consists of empty lists
    new_column = list(range(len(dataset))) 
    dataset=dataset.add_column('id',new_column)
    dataset=dataset.rename_column( "query","anchor")
    dataset=dataset.rename_column( "answer","positive")
    return dataset

trn=get_hn_dataset(train_dataset, corpus_dataset)
trn.to_json(f'../data/trn_with_hard_negatives.json',orient='records')

eval=get_hn_dataset(eval_dataset, corpus_dataset)
eval.to_json(f'../data/eval_with_hard_negatives.json',orient='records')

test=get_hn_dataset(test_dataset, corpus_dataset).to_json(f'../data/tst_with_hard_negatives.json',orient='records')

# #save it and reload it
# dataset.to_json(f'../data/trn_with_hard_negatives_HF_new.json',orient='records',lines=True)
# load_dataset("json", data_files="../data/trn_with_hard_negatives_HF_new.json", split="train")
# dataset