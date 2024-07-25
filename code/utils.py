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
def get_queries_and_relevant_docs(dataset):
    """
    Extracts queries and relevant documents from a dataset.

    Args:
        dataset (huggingface dataset):

    Returns:
        tuple: A tuple containing two dictionaries:
            - queries: A dictionary mapping query IDs to query anchors.
            - relevant_docs: A dictionary mapping query IDs to relevant documents.
              Each query ID is associated with a list of relevant document IDs, where the first ID is the query ID itself.
    """
    queries = dict(
        zip(dataset["id"], dataset["anchor"])
    )  
    relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
    for q_id in queries:
        relevant_docs[q_id] = [q_id]
    return queries, relevant_docs

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
 

def setup_logger(modelname, mode='a'):
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

    return logger

def log_execution_time(logger, startTime):
    elapsed_time = time.time()-startTime 
    minutes, seconds = divmod(elapsed_time, 60)
    logger.info(f"Execution time: {int(minutes)} minutes and {seconds:.2f} seconds\n")