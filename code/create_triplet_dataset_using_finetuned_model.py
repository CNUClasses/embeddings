from myimports import *
import utils as ut
from tqdm.auto import tqdm  # so we see progress bar
from tqdm import tqdm
import json
from numba import njit

LOGGER=None

class index_values():
    """
    A class that provides indexing functionality for values.

    Attributes:
        filename (str): The filename to save the indexer object.

    Methods:
        getval(index): Returns the string value corresponding to the given index.
        getindex(val): Returns the int index corresponding to the given value.
        set(strings): Sets the indexer based on the unique strings provided.
    """

    def __init__(self, filename=None):
        """
        Initializes an instance of the index_values class.

        Args:
            filename (str): The filename to save the indexer object.
        """
        self.filename = filename  # used to save this indexer object
        self.__load()

    def getval(self, index):
        """
        Returns the value corresponding to the given index.

        Args:
            index (int): The index to retrieve the value for.

        Returns:
            The value corresponding to the given index.
        """
        return self.indexer_reverse[index]
    
    def getindex(self, val):
        """
        Returns the index corresponding to the given value.

        Args:
            val: The value to retrieve the index for.

        Returns:
            The index corresponding to the given value.
        """
        return self.indexer[val]

    def set(self, strings):
        """
        Sets the indexer based on the unique strings provided.

        Args:
            strings (list): A list of strings to set the indexer with.
        """
        unique_strings = list(set(strings))
        self.indexer = {val: index for index, val in enumerate(unique_strings)}
        self.indexer_reverse = {index: val for val, index in self.indexer.items()}
        self.__store()

    def __load(self):
        """
        Loads the indexer from the specified file, if it exists.
        """
        if self.filename is None:
            return
        
        self.indexer = {}
        self.indexer_reverse = {}
        
        if os.path.exists(self.filename):
            with open(self.filename) as f:
                self.indexer = json.load(f)
                self.indexer_reverse = {index: val for val, index in self.indexer.items()}

    def __store(self):
        """
        Stores the indexer in the specified file.
        """
        if self.filename is None:
            return
        with open(self.filename, 'w') as f:
            json.dump(self.indexer, f)

def preprocess_text(text):
    """
    clean white space and lower case the text
    """
    return " ".join(text.split()).lower()

### Hard negative mine the positives for similar positives
### This assummes the model has been fine tuned on the dataset first
# Should I take absolute value of cosign similarity so it goes from 0-1?
#the following runs on GPU
def get_scores_processed(scores, high=0.65):
    """
    Process the scores and generate a mask based on a threshold.

    Parameters:
    scores (torch.Tensor): The input scores.
    high (float, optional): The threshold value. Defaults to 0.65.

    Returns:
    tuple: A tuple containing two numpy arrays - the processed scores and the mask.

    """
    
    # Get the entries below high
    scores_mask = (scores < high).to(scores.device)

    # Set diagonal to False (don't want true positive included in the hard negatives)
    # mask = (torch.eye(scores_mask.shape[0], scores_mask.shape[0]) > 0).to(scores.device)
    # scores_mask.masked_fill_(mask, False)
   
    return (scores.cpu().detach().numpy(), scores_mask.cpu().detach().numpy())


@njit
def get_hard_negatives_CPU(scores,scores_mask,positives, high=0.65,low=0.5, topn=5):
    """
    Train a sentencetransformer model, get its average similarity score, use range around that average for hard
    negatives
    expects scores to be nxn matrix of similarity scores, nparray
    expects positives to be a list of n strings
    expects high to be floats denoting the max acceptable similarity score
    topn: int, number of hard negatives to return from torch.top_k
    Get pairs of indices with low<= score <= high
    returns: list of list of positives whose similarity score is between low and high
    """
    
      # use scores_mask to select hard negatives
    hard_negatives=[]
    hn_found=0
    poor_hn_found=0

    # cntr=0
    for i,row in enumerate(scores_mask):
        # get all the positives and their scores
        # #make sure none of these positives are the same as the true positive
        # #this happens when you derive multiple queries from the same positive
        
        candidates=[(scores[i,j],positives[j]) for j in range(len(row)) if row[j]==True]     
 
        #sort by score      
        candidates.sort(key=lambda x: x[0],reverse=True)
 
        #find and remove all duplicates, do not include the true positive in candidates
        seen = set()   #empty
        seen.add(positives[i])  #do not want to return the true positive
        candidates= [tup for tup in candidates if not (tup[1] in seen or seen.add(tup[1]))]
        
        #we want the top n only so just consider those
        candidates=candidates[:topn]
        res = [pos for score, pos in candidates if score >= low ]

        if(len(res)==0):
            #nothing found between high and low, take the highest 1 seen and return it
            # res = [pos for score, pos in candidates[:1]]
            # if(len(res)==0):
            #     print(f'res=[] for row {i}')
            poor_hn_found+=1
        else:
            hn_found+=1
        
        hard_negatives.append(res)
        # cntr+=1
        # if(cntr%1000==0):
        #     print(f"{cntr} rows processed")
        
    print(f"hn_found={hn_found}, poor_hn_found={poor_hn_found}")
    return hard_negatives

def get_scores(modelname, all_positives, anchors, all_positives_embeddings=None):
    global LOGGER

    # Load pre-trained Sentence Transformer Model. It will be downloaded automatically
    LOGGER.info(f'### loading sentencetransformer {ut.modelname}')

    #expect a trained model to be in the models directory, the training will help with hard negative mining
    model = SentenceTransformer(f"./models/{modelname}",device="cuda:0" if torch.cuda.is_available() else "cpu",)

    # Use "convert_to_tensor=True" to keep the tensors on GPU (if available)
    if(all_positives_embeddings is None):   #calculate once
        all_positives_embeddings = model.encode(all_positives, convert_to_tensor=True)
    anchor_embeddings = model.encode(anchors, convert_to_tensor=True)

    LOGGER.info(f'### generating similarity score matrix')

    # We use cosine-similarity 
    scores=model.similarity(anchor_embeddings, all_positives_embeddings)

    #to save memory do the following
    del anchor_embeddings
    ut.clean_up()
    return scores,all_positives_embeddings

def get_df(dataset):
    df = pd.read_json(f'../data/{dataset}.json')
    df.drop_duplicates(subset=['anchor', 'positive'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    try:
        df.drop(columns=['most_dissimilar_context'],inplace=True)
    except:
        pass
    return df

def getsentencelists(df,col):
    '''
    param: df:pd dataframe
    param: col-column in dataframe to return lists from
    return: lists of all strings in column
     of InputExample objects
     ex.
     col='positive'
    positives= getsentencelists(df,col)
    ''' 
    #clean the text
    df[col] = df[col].apply(lambda x: preprocess_text(x))

    res=[]
    for _,row in tqdm(df.iterrows()):
       res.append(row[col])
    return res

# def get_indexer(all_positives,fn=None):
#     #save all positives to consider when doing hard negative mining
#     indexer=index_values(fn)
#     indexer.set(all_positives)
#     return indexer

def get_negatives(all_positives, scores,indexer=None,positives_index=None, high=.65, low=0.5, topn=3):
    global LOGGER
    LOGGER.info(f'### #convert all string values to their index for speed and space savings')
    if indexer is None:
        indexer=index_values(None)
        indexer.set(all_positives)
        positives_index=[indexer.getindex(val) for val in all_positives]

    #get the hard negatives
    processed_scores,mask=get_scores_processed(scores,high)
    ghn_int=get_hard_negatives_CPU(processed_scores,mask,positives_index,high=high,low=low,topn=topn)

    LOGGER.info('# NOTE!!!!! only saving the first out of the list for each row!')
    ghn_strs=[indexer.getval(val[0]) if len(val)>0 else '' for val in ghn_int ]
    return ghn_strs,indexer,positives_index

def main():
    '''to call this script
    python create_triplet_dataset_using_finetuned_model.py --high .95 --low .5  --mode a
    '''
    global LOGGER
    parser = argparse.ArgumentParser(description="Process some floating point numbers and a log filename.")
    
    parser.add_argument('--high', type=float, default=.95, help='a float value for high (default: .95)')
    parser.add_argument('--low', type=float, default=.60, help='a float value for low (default: .60)')
    # parser.add_argument('--log_fn', type=str, default='logfile.log', help='a log filename to record results (default: logfile.log)')
    parser.add_argument('--mode', type=str, choices=['a', 'a'], default='a', help='mode to open the log file: "a" for append, "w" for write/truncate (default: "a")')
    
    args = parser.parse_args()

    # what model are we using
    modelname=f"{ut.modelname.split('/')[-1]}"
    
    high=args.high
    low=args.low

    # Set up the LOGGER
    LOGGER = ut.setup_logger(modelname, args.mode)
    startTime = time.time()

    #suppress numba errors for logging
    from numba.core.errors import NumbaWarning
    import warnings
    warnings.simplefilter('ignore', category=NumbaWarning)

    LOGGER.info(f'### Hard negative mining for {modelname} with high={high}, low={low}')

    #datasets to get all positives from
    datasets=['trn','tst','eval']

    #get all positives from all datasets
    all_positives=[]
    for ds in datasets:
        #get the dataset
        df=get_df(ds)

        #save all positives to consider when doing hard negative mining
        positives=getsentencelists(df,'positive')
        all_positives.extend(positives)

    LOGGER.info(f'Got all positives from all datasets, total {len(all_positives)}\n')

    #calculate the following only once
    all_positives_embeddings=None
    indexer=None
    positives_index=None

    #datasets to calculate hard negatives for
    datasets=['trn', 'eval']
    for ds in datasets:
        #get the dataset
        df=get_df(ds)

        #get a list of anchors
        anchors=getsentencelists(df,'anchor')

        scores,all_positives_embeddings=get_scores(modelname, all_positives, anchors,all_positives_embeddings)
 
        #get the hard negatives, ignore the indexer
        ghn_strs,indexer,positives_index = get_negatives(all_positives, scores, indexer,positives_index, high=.95, low=0.5, topn=3)

        #save the hard negatives
        df['negative']=ghn_strs

        #save only rows that have a hard negative
        df=df[df['negative'].str.len()>0]

        df.reset_index(drop=True,inplace=True) #make sure the indexes are continuous
        
        #save to disk (!!use records or its loaded as 1 row in huggingface!)
        df.to_json(f'../data/{ds}_with_hard_negatives.json',orient="records")
        LOGGER.info(f'finished {ds} saved {len(df)} to ../data/{ds}_with_hard_negatives.json')

    ut.log_execution_time(LOGGER,startTime)
     
if __name__ == "__main__":
    main()

    