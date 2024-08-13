from myimports import *
import utils as ut
import chromadb
client = chromadb.Client()
from chromadb.utils import embedding_functions
LOGGER=None

#Lets see how it performs on multiple queries
from sentence_transformers import CrossEncoder
from tqdm import tqdm
import numpy as np
from collections import defaultdict

class track_stats:
    """
    A class to track statistics for document ranking.

    Attributes:
    - test_dataset (dict): The test dataset containing positive examples.
    - corpus_mapper (dict): A mapper to map document indices to their corresponding documents.
    - stats (defaultdict): A dictionary to store the statistics.

    Methods:
    - __init__(self, test_dataset, corpus_mapper): Initializes the track_stats object.
    - __call__(self, i, max_score): Updates the statistics based on the given index and maximum score.
    - print_stats(self): Prints the statistics.
    """

    def __init__(self, test_dataset, corpus_mapper, loss, crossencoder):
        self.test_dataset = test_dataset
        self.corpus_mapper = corpus_mapper
        self.stats = defaultdict(int) #defaults to 0
        self.loss = loss
        self.crossencoder=crossencoder

    def __call__(self, i, res_list):
        """
        Updates the statistics based on the given index and maximum score.

        Parameters:
        - i (int): The index of the example.
        - max_score (int): The maximum score obtained for the example.
        """  
        correct_doc = self.corpus_mapper[self.test_dataset['positive'][i]]
        original_choice = self.corpus_mapper[res_list[0]]
     
        self.stats['totals'] += 1
        docs=[self.corpus_mapper[res] for res in res_list]

        if correct_doc == original_choice:
            #top choice correct?
            self.stats['correct@1'] += 1
                       
        #in top 5?
        if(correct_doc in docs[:5]):
            self.stats['correct@5'] += 1

        #in top 10
        if(correct_doc in docs[:10]):
            self.stats['correct@10'] += 1
        
    def print_stats(self,logger):
        logger.info(f"Statistics for {self.loss} loss and {self.crossencoder} crossencoder")
        total=self.stats['totals']
        logger.info(f"self.stats['correct@1'] {self.stats['correct@1']} out of {total} for accuracy of {(self.stats['correct@1']/total)*100:.2f} %")
        logger.info(f"self.stats['correct@5'] {self.stats['correct@5']} out of {total} for accuracy of {(self.stats['correct@5']/total)*100:.2f} %")
        logger.info(f"self.stats['correct@10'] {self.stats['correct@10']} out of {total} for accuracy of {(self.stats['correct@10']/total)*100:.2f} %")
 

def main():
    '''to call this script
    python3 rerank_with_chromadb.py --mode a --localmodel y --loss triplet --modelname sentence-transformers/multi-qa-mpnet-base-cos-v1 --crossencoder 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    '''

    global LOGGER
    parser = argparse.ArgumentParser(description="rerank model outputs")
    # parser.add_argument('--log_fn', type=str, default='logfile.log', help='a log filename to record results (default: logfile.log)')
    parser.add_argument('--mode', type=str, choices=['a', 'w'], default='a', help='mode to open the log file: "a" for append, "w" for write/truncate (default: "a")')  
    parser.add_argument('--localmodel', type=str, choices=['y', 'n'], default='y', help='get model locally or from hugging face: "y" local, "n" hugging face (default: "y")')  
    parser.add_argument('--modelname', type=str, default='sentence-transformers/msmarco-distilbert-base-v2', help='which model to use(default: "sentence-transformers/msmarco-distilbert-base-v2")')  
    parser.add_argument('--crossencoder', type=str, default='sentence-transformers/msmarco-distilbert-base-v2', help='which model to use(default: "sentence-transformers/msmarco-distilbert-base-v2")')  
    parser.add_argument('--loss', type=str, choices=['MultipleNegativesRankingLoss', 'TripletLoss', 'CircleLoss','TripletLossOnlineHNMining','TripletLossOnlineSemiHNMining','GISTEmbedLoss' ],default='TripletLossOnlineSemiHNMining', help='loss function, CircleLoss and TripletLossOnlineHNMining are custom (default: "TripletLossOnlineSemiHNMining")')  
    parser.add_argument('--save_location', type=str, default=None, help='subdirectory where the final model is serialized. If none defaults to the name of the loss function. (default: None )')  

    argsp = parser.parse_args()

    #what model are we using
    modelname=f"{argsp.modelname.split('/')[-1]}"

    #where will the model be loaded/saved from/to?
    save_location=argsp.loss if argsp.save_location is None else argsp.save_location

    # 3. Load datasets
    train_dataset = load_dataset("json", data_files="../data/trn_with_hard_negatives.json", split="train")
    eval_dataset = load_dataset("json", data_files="../data/eval_with_hard_negatives.json", split="train")
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
 
    LOGGER.info(f"--RERANKING--- rerank_with_chromadb.py; model:{modelname}, reranker:{argsp.crossencoder}, localmodel:{argsp.localmodel}, loss function:{argsp.loss}")  

    if(argsp.localmodel=='n'):
        #get uploaded fine tuned embedder
        print("model from hugging face hub")
        st_ef=embedding_functions.SentenceTransformerEmbeddingFunction(f"kperkins411/{modelname}_{argsp.loss}_legal",device='cpu')
    else:
        #or from a local model
        print("model from local disk")
        st_ef=embedding_functions.SentenceTransformerEmbeddingFunction(f"./models/{modelname}/{save_location}/final",trust_remote_code=True,device=f"cuda:{ut.get_free_gpu()}" if torch.cuda.is_available() else "cpu",)
    
    # Create a new chroma collection
    st_collection = client.get_or_create_collection(name="st_embeddings", embedding_function=st_ef)

    #add all corpus values to collection
    st_collection.add(
        documents=list(corpus.values()),
        ids=[str(id) for id in list(corpus.keys())])
   
    #stat tracker    
    ts_original = track_stats(test_dataset,corpus_mapper, argsp.loss, argsp.crossencoder)
    ts_reranked = track_stats(test_dataset,corpus_mapper, argsp.loss+" reranked", argsp.crossencoder)

    #this is not fine tuned!  Also requires a http connection, so not good for a server that cycles every 40 minutes
    # RERANKER = CrossEncoder(argsp.crossencoder,num_labels = 1)

    # #test finetuned jobbie
    # model='cross-encoder/ms-marco-MiniLM-L-12-v2'
    # model_save_path = f"./models/finetuned_{model.replace('/','-')}"
    # RERANKER = CrossEncoder(model_save_path)

    #ragatouille gives worse scores on better models, not worth it
    # from ragatouille import RAGPretrainedModel
    # RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0", verbose=0)

    #Flag embeddings
    from FlagEmbedding import FlagReranker
    RERANKER = FlagReranker(argsp.crossencoder, use_fp16=True)
    # RERANKER = FlagReranker(argsp.crossencoder, )
 
    #get matches for all queries
    results = st_collection.query(
        # query_texts=test_dataset[0]['anchor'], #query single text
        query_texts=test_dataset['anchor'],  # Query all texts
        n_results=50    #10 results per query
    )

    # rerank the results with original query and documents returned from Chroma
    for i in tqdm(range(len(test_dataset))):
        #original
        res_list=results["documents"][i]
        ts_original(i, res_list)

        #reranked for standard rerankers
        scores = RERANKER.compute_score([(test_dataset['anchor'][i], doc) for doc in res_list])  #for Flag embeddings
        # scores = RERANKER.predict([(test_dataset['anchor'][i], doc) for doc in res_list])
        res_list_reranked=[x for _, x in sorted(zip(scores, results["documents"][i]), key=lambda pair: pair[0], reverse=True)]

        # ragatouille reranker, dont bother worse than ms_marco
        # res_list_reranked=[res['content'] for res in RERANKER.rerank(test_dataset['anchor'][i], res_list,k=10)]
        ts_reranked(i, res_list_reranked)
    
    ts_original.print_stats(LOGGER)
    ts_reranked.print_stats(LOGGER)

    ut.log_execution_time(LOGGER,startTime)

if __name__ == "__main__":
    main()
