from myimports import *

#hugging face login from environmental variable set in .bash_profile
from huggingface_hub import login
login(token=f"{os.environ.get('HUGGING_FACE_TOKEN')}", add_to_git_credential=True)  # ADD YOUR TOKEN HERE

# modelname='sentence-transformers/multi-qa-MiniLM-L6-dot-v1' #not normalized, suitable for dot product not cosign similarity 
# modelname='sentence-transformers/multi-qa-MiniLM-L6-cos-v1'  #cosign similarity
# batch_size=128

# modelname='sentence-transformers/msmarco-MiniLM-L6-cos-v5'
modelname='BAAI/bge-base-en-v1.5'
batch_size=64

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
