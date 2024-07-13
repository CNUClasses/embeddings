from myimports import *
import utils as ut
from tqdm.auto import tqdm  # so we see progress bar
from tqdm import tqdm
import json
from numba import njit
import logging

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
    mask = (torch.eye(scores_mask.shape[0], scores_mask.shape[0]) > 0).to(scores.device)
    scores_mask.masked_fill_(mask, False)
   
    return (scores.cpu().detach().numpy(), scores_mask.cpu().detach().numpy())


@njit
def get_hard_negatives_CPU(scores,scores_mask,positives,low=0.5, high=0.65, topn=5):
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
            res = [pos for score, pos in candidates[:1]]
            if(len(res)==0):
                print(f'res=[] for row {i}')
            poor_hn_found+=1
        else:
            hn_found+=1
        
        hard_negatives.append(res)
        # cntr+=1
        # if(cntr%1000==0):
        #     print(f"{cntr} rows processed")
        
    print(f"hn_found={hn_found}, poor_hn_found={poor_hn_found}")
    return hard_negatives

def get_scores(modelname, all_positives, anchors):
    # Load pre-trained Sentence Transformer Model. It will be downloaded automatically
    logger.info(f'### loading sentencetransformer {ut.modelname}')

    #expect a trained model to be in the models directory, the training will help with hard negative mining
    model = SentenceTransformer(f"./models/{modelname}",device="cuda:0" if torch.cuda.is_available() else "cpu",)

    # Use "convert_to_tensor=True" to keep the tensors on GPU (if available)
    positive_embeddings = model.encode(positives, convert_to_tensor=True)
    anchor_embeddings = model.encode(anchors, convert_to_tensor=True)

    logger.info(f'### generating similarity score matrix')

    # We use cosine-similarity 
    scores=model.similarity(anchor_embeddings, positive_embeddings)

    #to save memory do the following
    del positive_embeddings, anchor_embeddings
    ut.clean_up()
    return scores

def get_df(dataset):
    df = pd.read_json(f'../data/{dataset}.json')
    df.drop_duplicates(subset=['anchor', 'positive'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    #clean the text
    for col in cols:
        df.anchor = df[col].apply(lambda x: preprocess_text(x))
    
    try:
        df.drop(columns=['most_dissimilar_context'],inplace=True)
    except:
        pass
    return df

def getsentencelists(df,cols):
    '''
    param: df:pd dataframe
    param: cols list of columns in dataframe to return lists from
    return: tuple of lists, each list contains all strings in column
     of InputExample objects
     ex.
     cols=['positive','anchor']
    positives, anchors = getsentencelists(df,cols)
    ''' 
    
    res={}  
    for col in cols:
        res[col]=[]
    for _,row in tqdm(df.iterrows()):
        for col in cols:
            res[col].append(row[col])
    return (res[col] for col in cols)


def get_negatives(all_positives, anchors, high=.65, low=0.5, topn=3):
    """
    This function retrieves the hard negatives for a given set of positive samples.

    Args:
        all_positives (list): A list of all positive samples.
        anchors (list): A list of anchor samples.
        high (float, optional): The upper threshold for the scores. Defaults to 0.65.
        low (float, optional): The lower threshold for the scores. Defaults to 0.5.
        topn (int, optional): The number of hard negatives to retrieve. Defaults to 3.

    Returns:
        list: A list of hard negative samples, 1 for each anchor.
        indexer: An indexer object for converting string values to their index.

    """
    logger.info(f'### #convert all string values to their index for speed and space savings')
    indexer = index_values(None)
    indexer.set(all_positives)
    positives_index = [indexer.getindex(val) for val in all_positives]

    # get the hard negatives
    processed_scores, mask = get_scores_processed(scores, high)
    ghn_int = get_hard_negatives_CPU(processed_scores, mask, positives_index, high=high, low=0.5, topn=3)

    logger.info('# NOTE!!!!! only saving the first out of the list for each row!')
    ghn_strs = [indexer.getval(val[0]) for val in ghn_int]
    return ghn_strs, indexer

    df['negative'] = ghn_strs
    df.reset_index(drop=True, inplace=True)  # make sure the indexes are continuous
def get_negatives(all_positives, anchors, high=.65, low=0.5, topn=3):
    logger.info(f'### #convert all string values to their index for speed and space savings')
    indexer=index_values(None)
    indexer.set(all_positives)
    positives_index=[indexer.getindex(val) for val in all_positives]

    #get the hard negatives
    processed_scores,mask=get_scores_processed(scores,high)
    ghn_int=get_hard_negatives_CPU(processed_scores,mask,positives_index,high=high,low=0.5,topn=3)

    logger.info('# NOTE!!!!! only saving the first out of the list for each row!')
    ghn_strs=[indexer.getval(val[0]) for val in ghn_int]
    return ghn_strs,indexer

    


if __name__ == "__main__":
    #suppress numba errors for logging
    from numba.core.errors import NumbaWarning
    import warnings
    warnings.simplefilter('ignore', category=NumbaWarning)

    startTime = datetime.now()

    # what model are we using
    modelname=f"{ut.modelname.split('/')[-1]}"

    #setup logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f"LOG_{modelname}_mine_hard_negatives.log", encoding='utf-8', level=logging.DEBUG, filemode="w",)

    #datasets to process
    datasets=['trn','tst','eval']

    #get all positives from all datasets
    all_positives=[]
    for ds in datasets:
        #save all positives to consider when doing hard negative mining
        positives=getsentencelists(ds,['positive'])
        all_positives.extend(positives)

    for ds in datasets:
        #get the dataset
        df=get_df(ds)

        #get a list of anchors
        anchors=getsentencelists(df,['anchor'])

        scores=get_scores(modelname, all_positives, anchors)

        #get the hard negatives, ignore the indexer
        ghn_strs,_ = get_negatives(all_positives, anchors, high=.65, low=0.5, topn=3)

        #save the hard negatives
        df['negative']=ghn_strs
        df.reset_index(drop=True,inplace=True) #make sure the indexes are continuous
        
        #save to disk (!!use records or its loaded as 1 row in huggingface!)
        df.to_json(f'../data/{dataset}_with_hard_negatives.json',orient="records")

    logger.info(f"--------- Script took  {datetime.now()-startTime} to run")




# def save_hard_negatives(dataset, ghn_strs):
#     try:
#         #get rid of rubbish is there
#         df.drop(columns=['most_dissimilar_context'],inplace=True)
#     except:
#         pass

#     #save to disk (!!use records or its loaded as 1 row in huggingface!)
#     df.to_json(f'../data/{dataset}_with_hard_negatives.json',orient="records")
#     return 
 
    # df = pd.read_json(f'../data/{dataset}.json')
    # df.drop_duplicates(subset=['anchor', 'positive'], inplace=True)
    # df.reset_index(drop=True, inplace=True)
    # df.anchor = df['anchor'].apply(lambda x: preprocess_text(x))
    # df.positive = df['positive'].apply(lambda x: preprocess_text(x))

    # # #### get the columns of interest
    # logger.info('### getting the columns of interest')
  
    # # print(f'There are {len(df)} rows but the positive column has only {df['positive'].nunique()} unique values')

    # #### generate the embeddings
    # logger.info('### generating the embeddings')

    # # Load pre-trained Sentence Transformer Model. It will be downloaded automatically
    # logger.info(f'### loading sentencetransformer {ut.modelname}')

    # #expect a trained model to be in the models directory, the training will help with hard negative mining
    # model = SentenceTransformer(f"./models/{modelname}",device="cuda:0" if torch.cuda.is_available() else "cpu",)

    # # Use "convert_to_tensor=True" to keep the tensors on GPU (if available)
    # positive_embeddings = model.encode(positives, convert_to_tensor=True)
    # anchor_embeddings = model.encode(anchors, convert_to_tensor=True)

    # logger.info(f'### generating similarity score matrix')

    # # We use cosine-similarity 
    # scores=model.similarity(anchor_embeddings, positive_embeddings)

    # #to save memory do the following
    # del positive_embeddings, anchor_embeddings
    # ut.clean_up()

    # logger.info(f'###save for ease of debugging')
    # torch.save(scores,'scores.pt')

    # import pickle
    # with open("positives", "wb") as fp:   #Pickling
    #     pickle.dump(positives, fp)

    # logger.info(f'### #convert all string values to their index to save space and speed this up')
    # indexer=index_values(None)
    # indexer.set(positives)
    # positives_index=[indexer.getindex(val) for val in positives]

    # #get the hard negatives
    # positives_index=[indexer.getindex(val) for val in positives]
    # vals=get_scores_processed(scores,high)
    # ghn_int=get_hard_negatives_CPU(*(vals),positives_index,high=high,low=0.5,topn=3)

    # logger.info('# NOTE!!!!! only saving the first out of the list for each row!')
    # ghn_strs=[indexer.getval(val[0]) for val in ghn_int]

    # df['negative']=ghn_strs
    # df.reset_index(drop=True,inplace=True) #make sure the indexes are continuous

    # try:
    #     df.drop(columns=['most_dissimilar_context'],inplace=True)
    # except:
    #     pass

    # #save to disk (!!use records or its loaded as 1 row in huggingface!)
    # df.to_json(f'../data/{dataset}_with_hard_negatives.json',orient="records")
    # return 

