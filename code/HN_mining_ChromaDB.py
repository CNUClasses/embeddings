#easy way run this BEFORE you import torch to select a particular device
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2"

# or this way, get any free GPU
from utils_gpu import get_free_gpu
get_free_gpu()

from myimports import *
import utils as ut
import chromadb
client = chromadb.Client()
from chromadb.utils import embedding_functions
import math
LOGGER=None

#Lets see how it performs on multiple queries
from tqdm import tqdm
import numpy as np

def main():
    '''to call this script
    python3 HN_mining.py --mode a --localmodel y --loss triplet --modelname sentence-transformers/multi-qa-mpnet-base-cos-v1 --crossencoder 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    '''
    global LOGGER
    parser = argparse.ArgumentParser(description="rerank model outputs")
    parser.add_argument('--mode', type=str, choices=['a', 'w'], default='a', help='mode to open the log file: "a" for append, "w" for write/truncate (default: "a")')  
    parser.add_argument('--localmodel', type=str, choices=['y', 'n'], default='y', help='get model locally or from hugging face: "y" local, "n" hugging face (default: "y")')  
    parser.add_argument('--modelname', type=str, default='sentence-transformers/multi-qa-mpnet-base-cos-v1', help='which model to use(default: "sentence-transformers/multi-qa-mpnet-base-cos-v1")')  
    parser.add_argument('--loss', type=str, choices=['MultipleNegativesRankingLoss', 'TripletLoss', 'CircleLoss','TripletLossOnlineHNMining','TripletLossOnlineSemiHNMining','GISTEmbedLoss' ,'CachedMultipleNegativesRankingLoss'],default='MultipleNegativesRankingLoss', help='loss function, CircleLoss and TripletLossOnlineHNMining are custom (default: "TripletLossOnlineSemiHNMining")')  
    parser.add_argument('--numb_HN_per_line', type=str, default='15', help='number of hard negatives to extract per line. (default:15)')  
    parser.add_argument('--save_location', type=str, default=None, help='subdirectory where the final model is serialized. If none defaults to the name of the loss function. (default: None )')  

    argsp = parser.parse_args()

    #what model are we using
    modelname=f"{argsp.modelname.split('/')[-1]}"

    #where will the model be loaded/saved from/to?
    save_location=argsp.loss if argsp.save_location is None else argsp.save_location

    # 3. Load datasets
    train_dataset = load_dataset("json", data_files="../data/trn.json", split="train")
    eval_dataset = load_dataset("json", data_files="../data/eval.json", split="train")
    test_dataset = load_dataset("json", data_files="../data/tst.json", split="train")

    # generate data for informationretreival evaluator
    corpus_dataset,corpus_mapper=ut.get_corpus_and_corpus_mapper(train_dataset, eval_dataset, test_dataset, dup_col='positive')

    #collect all positives from train,eval,test
    corpus = dict(
        zip(corpus_dataset["id"], corpus_dataset["positive"])
    )  # Our corpus (cid => document)

     # Set up the LOGGER
    LOGGER = ut.setup_logger(modelname, argsp.mode)
    startTime = time.time()
 
    LOGGER.info(f"--Mining Hard Negatives Corpus using chromadb; model:{modelname}, localmodel:{argsp.localmodel}, loss function:{argsp.loss}--")  

    if(argsp.localmodel=='n'):
        #get uploaded fine tuned embedder
        print("model from hugging face hub")
        st_ef=embedding_functions.SentenceTransformerEmbeddingFunction(f"kperkins411/{modelname}_{argsp.loss}_legal",trust_remote_code=True, device='cuda')
    else:
        #or from a local model
        print("model from local disk")
        st_ef=embedding_functions.SentenceTransformerEmbeddingFunction(f"./models/{modelname}/{save_location}/final",trust_remote_code=True, device='cuda')
  
        
    # Create a new chroma collection
    st_collection = client.get_or_create_collection(name="st_embeddings",metadata={"hnsw:space": "cosine"}, embedding_function=st_ef)

    #add all corpus values to collection to embed
    st_collection.add(
        documents=list(corpus.values()),
        ids=[str(id) for id in list(corpus.keys())])
 
    #get matches for each dataset of interest
    def get_HNs(st_collection,ds, numb_hard_negatives_per_line):
        """
        Retrieves a list of hard negatives (HNs) for each line in the dataset.
        Args:
            st_collection (object): The collection of embeddings to search.
            ds (dict): The dataset containing the anchor and positive texts.
            numb_hard_negatives_per_line (int): The number of hard negatives to include per line.
        Returns:
            list: A list of hard negatives for each line in the dataset.
        """

        # find the closest documents to each anchor in ds
        total_to_retreive=200
        t5=int(total_to_retreive/20)   # top 5%  
        t25=int(total_to_retreive/4)   # top 25% 

        #get 1/5 hard negatives from top 5% and 1/5 from 5%-25% and 3/5 from 25%-100%
        numb_HN=int(numb_hard_negatives_per_line/5)

        res= st_collection.query(query_texts=ds['anchor'],n_results=total_to_retreive) 
        res=res['documents']

        for i in tqdm(range(len(ds))):
            #remove the correct positive for each line
            try:
                res[i].remove(ds['positive'][i]) #??
            except:
                pass

            #Sample hard negatives
            #numb_hard_negatives_per_line from first from top 5% (Hardest negatives)
            #numb_hard_negatives_per_line from 5%-25% (easier negatives)
            #numb_easy_negatives_per_line from 25%-100% (easiest negatives)
            res[i]=random.sample(res[i][:t5], numb_HN)+random.sample(res[i][t5:t25], numb_HN) + random.sample(res[i][t25:total_to_retreive], 3*numb_HN)

        #add a negative column to ds
        ds=ds.add_column('neg',res)
        return ds
    
    #get the number of hard and easy negatives to mine per line
    total_negatives_per_line=int(argsp.numb_HN_per_line)   
 
    #get the hard negatives
    train_dataset=get_HNs(st_collection,train_dataset,numb_hard_negatives_per_line=total_negatives_per_line)
    eval_dataset=get_HNs(st_collection,eval_dataset,numb_hard_negatives_per_line=total_negatives_per_line)
 
    #save the dataset with hard negatives
    #TODO this is very ineffecient, should have a lookup table and numbers for each positive and negative
    train_dataset.to_json(f"../data/trn_HN_KP.json",orient='records',lines=True)
    eval_dataset.to_json(f"../data/eval_HN_KP.json",orient='records',lines=True)

    def fixup(df):
        """
        Explodes the 'neg' column in the given DataFrame and renames the columns.
        
        Args:
            df (pandas.DataFrame): The input DataFrame.
            
        Returns:
            pandas.DataFrame: The modified DataFrame with exploded 'neg' column and renamed columns.
        """
        df=df.explode('neg')
        df.rename(columns={'query':'anchor','pos':'positive','neg':'negative'}, inplace=True)
        id=range(len(df))
        df['id']=id
        return df
    

   #Load the dataset, then fixup, then save
    trn=pd.read_json("../data/trn_HN_KP.json", orient="records",lines=True)
    eval=pd.read_json("../data/eval_HN_KP.json", orient="records",lines=True)
 
    trn=fixup(trn)
    eval=fixup(eval)

    trn.to_json("../data/trn_HN_KP.json", orient="records",lines=True)
    eval.to_json("../data/eval_HN_KP.json", orient="records",lines=True)

    ut.log_execution_time(LOGGER,startTime)

if __name__ == "__main__":
    main()
