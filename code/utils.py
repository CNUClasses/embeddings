from myimports import *

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
transformers.set_seed(42)

#hugging face login from environmental variable set in .bash_profile
def login_hf():
    from huggingface_hub import login
    login(token=f"{os.environ.get('HUGGING_FACE_TOKEN')}", add_to_git_credential=True)  # ADD YOUR TOKEN HERE

# modelname='sentence-transformers/multi-qa-MiniLM-L6-dot-v1' #not normalized, suitable for dot product not cosign similarity 
# modelname='sentence-transformers/multi-qa-MiniLM-L6-cos-v1'  #cosign similarity
# batch_size=128

# modelname='sentence-transformers/msmarco-MiniLM-L6-cos-v5'
# modelname='BAAI/bge-large-en-v1.5'   #does not work well with leagal dataset and 4 epochs, see log file
# modelname='msmarco-distilbert-base-dot-prod-v3'
# modelname='msmarco-MiniLM-L-6-v3'

# model = SentenceTransformer("intfloat/e5-mistral-7b-instruct")
# batch_size=32

# modelname='sentence-transformers/msmarco-distilbert-base-v2'
# batch_size=128

# modelname='sentence-transformers/multi-qa-mpnet-base-cos-v1'
# batch_size=64

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
    if(type(col)==list):
        df = df.drop_duplicates(subset=col)
    else:
        df = df.drop_duplicates(subset=[col])
    df.reset_index(drop=True, inplace=True)
    if(verbose==True):
        print(f'Length ds before dropping duplicates:{nr} rows, dropped {nr - len(df)} duplicate rows. {len(df)} rows remain')
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

    ds = pd.DataFrame(corpus_dataset)
    ds = drop_duplicate_rows(ds, dup_col,verbose)  # Drop all rows that have duplicates in the positive column
    corpus_dataset = datasets.Dataset.from_pandas(ds, preserve_index=False)

    # Create a dictionary mapping positive values to their corresponding IDs
    corpus_mapper = dict(zip(corpus_dataset['positive'], corpus_dataset['id']))

    return corpus_dataset, corpus_mapper

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

    # log_filename=getlogfile(modelname, mode)
    log_filename=f"./logs/LOG_{modelname}.log"

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

def log_performance(res, logger, modelname:str, loss:str, info=""):
    '''
    logs performance metrics
    '''
    logger.info(f"{info}:  model:{modelname} loss:{loss}")
    logger.info(f"cosine_ndcg@10    : {res[modelname+'_cosine_ndcg@10']:.2f}")
    logger.info(f"cosine_mrr@10     : {res[modelname+'_cosine_mrr@10']:.2f}")
    logger.info(f"cosine_map@100    : {res[modelname+'_cosine_map@100']:.2f}")
    logger.info(f"cosine_accuracy@1 : {res[modelname+'_cosine_accuracy@1']:.2f}")
    logger.info(f"cosine_accuracy@5 : {res[modelname+'_cosine_accuracy@5']:.2f}")
    logger.info(f"cosine_accuracy@10: {res[modelname+'_cosine_accuracy@10']:.2f}\n")

# EDIT 10/04/2022 - This version was provided by @jayelm who fixed some bugs and made the function much more robust

import os
import subprocess
import time

def get_free_gpu(threshold_vram_usage=1500, max_gpus=1, wait=False, sleep_time=10):
    """
    Assigns free gpus to the current process via the CUDA_AVAILABLE_DEVICES env variable
    This function should be called after all imports,
    in case you are setting CUDA_AVAILABLE_DEVICES elsewhere
    Borrowed and fixed from https://gist.github.com/afspies/7e211b83ca5a8902849b05ded9a10696
    Args:
        threshold_vram_usage (int, optional): A GPU is considered free if the vram usage is below the threshold
                                              Defaults to 1500 (MiB).
        max_gpus (int, optional): Max GPUs is the maximum number of gpus to assign.
                                  Defaults to 2.
        wait (bool, optional): Whether to wait until a GPU is free. Default False.
        sleep_time (int, optional): Sleep time (in seconds) to wait before checking GPUs, if wait=True. Default 10.
    """

    def _check():
        # Get the list of GPUs via nvidia-smi
        smi_query_result = subprocess.check_output(
            "nvidia-smi -q -d Memory | grep -A4 GPU", shell=True
        )
        # Extract the usage information
        gpu_info = smi_query_result.decode("utf-8").split("\n")
        gpu_info = list(filter(lambda info: "Used" in info, gpu_info))
        gpu_info = [
            int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_info
        ]  # Remove garbage
        # Keep gpus under threshold only
        free_gpus = [
            str(i) for i, mem in enumerate(gpu_info) if mem < threshold_vram_usage
        ]
        free_gpus = free_gpus[: min(max_gpus, len(free_gpus))]
        gpus_to_use = ",".join(free_gpus)
        return gpus_to_use

    while True:
        gpus_to_use = _check()
        if gpus_to_use or not wait:
            break
        print(f"No free GPUs found, retrying in {sleep_time}s")
        time.sleep(sleep_time)

    if not gpus_to_use:
        raise RuntimeError("No free GPUs found")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus_to_use
        # logger.info(f"Using GPU(s): {gpus_to_use}")
        print(f"Using GPU(s): {gpus_to_use}")
        return gpus_to_use
    