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
    # parser.add_argument('--log_fn', type=str, default='logfile.log', help='a log filename to record results (default: logfile.log)')
    parser.add_argument('--mode', type=str, choices=['a', 'w'], default='a', help='mode to open the log file: "a" for append, "w" for write/truncate (default: "a")')  
    parser.add_argument('--localmodel', type=str, choices=['y', 'n'], default='y', help='get model locally or from hugging face: "y" local, "n" hugging face (default: "y")')  
    parser.add_argument('--modelname', type=str, default='sentence-transformers/multi-qa-mpnet-base-cos-v1', help='which model to use(default: "sentence-transformers/multi-qa-mpnet-base-cos-v1")')  
    parser.add_argument('--loss', type=str, choices=['MultipleNegativesRankingLoss', 'TripletLoss', 'CircleLoss','TripletLossOnlineHNMining','TripletLossOnlineSemiHNMining','GISTEmbedLoss' ,'CachedMultipleNegativesRankingLoss'],default='MultipleNegativesRankingLoss', help='loss function, CircleLoss and TripletLossOnlineHNMining are custom (default: "TripletLossOnlineSemiHNMining")')  
    parser.add_argument('--numb_HN_per_line', type=str, default='15', help='number of hard negatives to extract per line. (default:15)')  
    # parser.add_argument('--use_random_sample', type=str, choices=['y', 'n'], default='y', help='randomly sample from closest hard negatives. (default:y)')  
    parser.add_argument('--fraction_HN_to_semiHN', type=str, default='0.2', help='ratio of the number of hard negatives to semi hard negatives, .1 means 1 HN to 10 semiHN. (default:.1)')  
 
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
        st_ef=embedding_functions.SentenceTransformerEmbeddingFunction(f"kperkins411/{modelname}_{argsp.loss}_legal",device='cpu')
    else:
        #or from a local model
        print("model from local disk")
        st_ef=embedding_functions.SentenceTransformerEmbeddingFunction(f"./models/{modelname}/{save_location}/final",trust_remote_code=True,device=f"cuda:{ut.get_free_gpu()}" if torch.cuda.is_available() else "cpu",)
        
        #see if model makes better HNs (performs same as msmarco-distilbert-cos-v5)
        # st_ef=embedding_functions.SentenceTransformerEmbeddingFunction(f"BAAI/bge-base-en-v1.5",trust_remote_code=True,device=f"cuda:{ut.get_free_gpu()}" if torch.cuda.is_available() else "cpu",)
 
        
    # Create a new chroma collection
    st_collection = client.get_or_create_collection(name="st_embeddings",metadata={"hnsw:space": "cosine"}, embedding_function=st_ef)

    #add all corpus values to collection to embed
    st_collection.add(
        documents=list(corpus.values()),
        ids=[str(id) for id in list(corpus.keys())])
 
    #get matches for each dataset of interest
    # def get_HNs(st_collection,ds,numb_easy_negatives_per_line, numb_hard_negatives_per_line, use_random_sample=False):
        
    #     all=[]
    #     for i in tqdm(range(len(ds))):
    #         res= st_collection.query(query_texts=ds['anchor'][i],n_results=100 ) 
    #         res=res['documents'][0] 

    #         #remove the actual positive for each line
    #         # print(res.documents[i])
    #         try:
    #             res.remove(ds['positive'][i]) #??
    #         except:
    #             pass

    #         #first get the hardest negatives, then a random sample from the easier ones
    #         res1=random.sample(res[:10], numb_hard_negatives_per_line)+random.sample(res[10:50], numb_hard_negatives_per_line) + random.sample(res[50:100], numb_easy_negatives_per_line)
    #         all.append(res1)
    #         # res['documents'][i]=res['documents'][i][:numb_hard_negatives_per_line] + random.sample(res['documents'][i][100:200], numb_easy_negatives_per_line)

    #         #then randomly sample the rest from the 

    #         # #return a subset of the data?
    #         # if(use_random_sample=='y'):
    #         #     res['documents'][i]=list(np.random.choice(res['documents'][i],numb_HN_per_line,replace=False))
    #         # else:
    #         #     # print(res[i].documents)
    #         #     res['documents'][i]=res['documents'][i][:numb_HN_per_line]
    #     return all
    def get_HNs(st_collection,ds,numb_easy_negatives_per_line, numb_hard_negatives_per_line, use_random_sample=False):
            
        res= st_collection.query(query_texts=ds['anchor'],n_results=100 ) 
        res=res['documents']

        for i in tqdm(range(len(ds))):
            # res= st_collection.query(query_texts=ds['anchor'],n_results=100 ) 
            # res=res['documents'][0] 

            #remove the actual positive for each line
            # print(res.documents[i])
            try:
                res[i].remove(ds['positive'][i]) #??
            except:
                pass

            #first get the hardest negatives, then a random sample from the easier ones
            res[i]=random.sample(res[i][:10], numb_hard_negatives_per_line)+random.sample(res[i][10:50], numb_hard_negatives_per_line) + random.sample(res[i][50:100], numb_easy_negatives_per_line)
            # res['documents'][i]=res['documents'][i][:numb_hard_negatives_per_line] + random.sample(res['documents'][i][100:200], numb_easy_negatives_per_line)

            #then randomly sample the rest from the 

            # #return a subset of the data?
            # if(use_random_sample=='y'):
            #     res['documents'][i]=list(np.random.choice(res['documents'][i],numb_HN_per_line,replace=False))
            # else:
            #     # print(res[i].documents)
            #     res['documents'][i]=res['documents'][i][:numb_HN_per_line]
        return res
    #get the number of hard and easy negatives to mine per line
    total_negatives_per_line=int(argsp.numb_HN_per_line)   
    numb_hard_negatives_per_line=math.ceil(total_negatives_per_line*float(argsp.fraction_HN_to_semiHN))
    numb_easy_negatives_per_line=total_negatives_per_line-2*numb_hard_negatives_per_line

    # res_trn=get_HNs(st_collection,train_dataset,numb_easy_negatives_per_line=numb_easy_negatives_per_line,numb_hard_negatives_per_line=numb_hard_negatives_per_line, use_random_sample=argsp.use_random_sample)
    # res_eval=get_HNs(st_collection,eval_dataset,numb_easy_negatives_per_line=numb_easy_negatives_per_line,numb_hard_negatives_per_line=numb_hard_negatives_per_line, use_random_sample=argsp.use_random_sample)

    res_trn=get_HNs(st_collection,train_dataset,numb_easy_negatives_per_line=numb_easy_negatives_per_line,numb_hard_negatives_per_line=numb_hard_negatives_per_line)
    res_eval=get_HNs(st_collection,eval_dataset,numb_easy_negatives_per_line=numb_easy_negatives_per_line,numb_hard_negatives_per_line=numb_hard_negatives_per_line)

    train_dataset=train_dataset.add_column('neg',res_trn['documents'])
    eval_dataset=eval_dataset.add_column('neg',res_eval['documents'])
 
    #save the dataset with hard negatives
    #TODO this is very ineffecient, should have a lookup table and numbers for each positive and negative
    train_dataset.to_json(f"../data/trn_HN_KP.json",orient='records',lines=True)
    eval_dataset.to_json(f"../data/eval_HN_KP.json",orient='records',lines=True)

    #Lets explode the negative column
    trn=pd.read_json("../data/trn_HN_KP.json", orient="records",lines=True)
    eval=pd.read_json("../data/eval_HN_KP.json", orient="records",lines=True)

    def fixup(df):
        df=df.explode('neg')
        df.rename(columns={'query':'anchor','pos':'positive','neg':'negative'}, inplace=True)
        id=range(len(df))
        df['id']=id
        return df

    trn=fixup(trn)
    eval=fixup(eval)

    trn.to_json("../data/trn_HN_KP.json", orient="records",lines=True)
    eval.to_json("../data/eval_HN_KP.json", orient="records",lines=True)

    ut.log_execution_time(LOGGER,startTime)

if __name__ == "__main__":
    main()
