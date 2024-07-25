from myimports import *

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
transformers.set_seed(42)

#hugging face login from environmental variable set in .bash_profile
from huggingface_hub import login
login(token=f"{os.environ.get('HUGGING_FACE_TOKEN')}", add_to_git_credential=True)  # ADD YOUR TOKEN HERE

# modelname='sentence-transformers/multi-qa-MiniLM-L6-dot-v1' #not normalized, suitable for dot product not cosign similarity 
# modelname='sentence-transformers/multi-qa-MiniLM-L6-cos-v1'  #cosign similarity
# batch_size=128

# modelname='sentence-transformers/msmarco-MiniLM-L6-cos-v5'
modelname='BAAI/bge-large-en-v1.5'   #does not work well with leagal dataset and 4 epochs, see log file
# modelname='msmarco-distilbert-base-dot-prod-v3'
# modelname='msmarco-MiniLM-L-6-v3'
modelname='sentence-transformers/msmarco-distilbert-base-v2'

batch_size=128

import torch, gc
def clean_up():
    gc.collect()
    torch.cuda.empty_cache()
clean_up()

def drop_duplicate_rows(df, col,verbose=True):
    """
    Remove duplicate rows from a DataFrame based on a specified column.

    Args:
        df (pandas.DataFrame): The DataFrame to remove duplicates from.
        col (str): The column to check for duplicates.

    Returns:
        pandas.DataFrame: The DataFrame with duplicate rows removed.

    Example:
        df = pd.DataFrame({'A': [1, 2, 3, 3], 'B': [4, 5, 6, 6]})
        df = drop_duplicate_rows(df, 'A')
        print(df)
        # Output:
        #    A  B
        # 0  1  4
        # 1  2  5
        # 2  3  6
    """
    nr = len(df)
    df = df.drop_duplicates(subset=[col])
    df.reset_index(drop=True, inplace=True)
    if(verbose==True):
        print(f'dropped {nr - len(df)} rows. have {len(df)} rows left')
    return df

#to convert dataset column type
import datasets as ds
def change_col_to_list(df,col):
    '''
    takes col in df and changes its type to a list
    expects a single string in col

    ex:
    df1 = ds.Dataset.from_dict({"column_1": ['a 1','b 1']})
    df1 = df1.add_column("column_2", ['c 2','d 2'])
    df1=change_col_to_list(df1,'column_1')
    df1.to_json(f'test2.jsonl',orient='records',lines=True)

    results in  test2.jsonl
    {"column_2":"c 2","column_1":["a 1"]}
    {"column_2":"d 2","column_1":["b 1"]}

    '''
    #copy orig values
    data=df[col]
    data=[[val] for val in data]

    #delete orig column
    df=df.remove_columns([col])

    #add back as list
    df = df.add_column(col, data)

    return df

#### training stuff#####
# def get_queries_and_relevant_docs(dataset,mapper):
#     """
#     Extracts queries and relevant documents from a dataset.

#     Args:
#         dataset (huggingface dataset):
#         mapper (dict): A dictionary mapping a string to its associated ID in the corpus.

#     Returns:
#         tuple: A tuple containing two dictionaries:
#             - queries: A dictionary mapping query IDs to query anchors.
#             - relevant_docs: A dictionary mapping query IDs to relevant documents.
#               Each query ID is associated with a list of relevant document IDs, where the first ID is the query ID itself.
#     """
#     queries = dict(
#         zip(dataset["id"], dataset["anchor"])
#     )  
#     relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
#     for q_id in queries:
#         relevant_docs[q_id] = [q_id]
#     return queries, relevant_docs
def get_queries_and_relevant_docs(dataset,mapper):
    """
    Extracts queries and relevant documents from a dataset.
    handles the case of duplicate positives.

    Args:
        dataset (huggingface dataset):
        mapper (dict): A dictionary mapping a string to its associated ID in the corpus.

    Returns:
        tuple: A tuple containing two dictionaries:
            - queries: A dictionary mapping query IDs to query anchors.
            - relevant_docs: A dictionary mapping query IDs to relevant documents.
              Each query ID is associated with a list of relevant document IDs, where the first ID is the query ID itself.

    usage:
    corpus_dataset = concatenate_datasets([train_dataset, eval_dataset, test_dataset])
    #drop duplicates
    ds = pd.DataFrame(corpus_dataset)
    ds=ut.drop_duplicate_rows(ds,'positive')  #dump all rows that have duplicates in the positive column
    corpus_dataset = datasets.Dataset.from_pandas(ds, preserve_index=False)
    corpus_mapper=dict(zip(corpus_dataset['positive'],corpus_dataset['id']))

    eval_queries, eval_relevant_docs=get_queries_and_relevant_docs(eval_dataset,corpus_mapper)

    """
    queries = dict(
        zip(dataset["id"], dataset["anchor"])
    )  
    docs=dict(
        zip(dataset["id"], dataset["positive"])
    )
    relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
    for q_id in queries:
        relevant_docs[q_id] = [mapper[docs[q_id]]]
    return queries, relevant_docs

def get_corpus_and_corpus_mapper(trn:Dataset, eval:Dataset, tst:Dataset, dup_col='positive',verbose=True):
    """
    Concatenates the training, evaluation, and test datasets into a single corpus dataset.
    Drops duplicate rows based on the specified column.
    Returns the corpus dataset and a dictionary mapping positive values to their corresponding IDs.

    Args:
        trn (Dataset): The training dataset.
        eval (Dataset): The evaluation dataset.
        tst (Dataset): The test dataset.
        dup_col (str, optional): The column to check for duplicates. Defaults to 'positive'.

    Returns:
        corpus_dataset (Dataset): The concatenated corpus dataset.
        corpus_mapper (dict): A dictionary mapping positive values to their corresponding IDs.
    """

    # Concatenate the datasets
    corpus_dataset = concatenate_datasets([trn, eval, tst])

    # Drop duplicates
    if(verbose==True):
        print(f'len(corpus_dataset) before dropping duplicates:{len(corpus_dataset)}')

    ds = pd.DataFrame(corpus_dataset)
    ds = drop_duplicate_rows(ds, dup_col,verbose)  # Drop all rows that have duplicates in the positive column
    corpus_dataset = datasets.Dataset.from_pandas(ds, preserve_index=False)

    if(verbose==True):
        print(f'len(corpus_dataset) after dropping duplicates:{len(corpus_dataset)}')

    # Create a dictionary mapping positive values to their corresponding IDs
    corpus_mapper = dict(zip(corpus_dataset['positive'], corpus_dataset['id']))

    return corpus_dataset, corpus_mapper

def get_corpus_and_corpus_mapper(trn:Dataset,eval:Dataset,tst:Dataset, dup_col='positive'):

    corpus_dataset = concatenate_datasets([trn, eval, tst])
    #drop duplicates
    ds = pd.DataFrame(corpus_dataset)
    ds=drop_duplicate_rows(ds,dup_col)  #dump all rows that have duplicates in the positive column
    corpus_dataset = datasets.Dataset.from_pandas(ds, preserve_index=False)
    corpus_mapper=dict(zip(corpus_dataset['positive'],corpus_dataset['id']))

    return corpus_dataset,corpus_mapper

def getlogfile(modelname, mode):
    '''
    returns a log file name
    '''
    #find the next available new log file name
    i=0
    while os.path.exists(f"./logs/LOG_{modelname}_{i}.log"):
        i += 1

    #if appending, get last logfile for this model
    if mode=='a':
        i=i-1
    print(f"Logging to: ./logs/LOG_{modelname}_{i}.log, mode={mode}")
    return f"./logs/LOG_{modelname}_{i}.log"
 

def setup_logger(modelname, mode='a', verbose=True):
    # Create a logger object
    logger = logging.getLogger('custom_logger')
    logger.setLevel(logging.INFO)

    log_filename=getlogfile(modelname, mode)

    # Create a file handler which logs messages to a file
    fh = logging.FileHandler(log_filename, mode=mode)
    fh.setLevel(logging.INFO)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(fh)

    #add logging to cout
    if (verbose == True):
        logger.addHandler(logging.StreamHandler(sys.stdout))

    return logger

def log_execution_time(logger, startTime):
    elapsed_time = time.time()-startTime 
    minutes, seconds = divmod(elapsed_time, 60)
    logger.info(f"Execution time: {int(minutes)} minutes and {seconds:.2f} seconds\n")