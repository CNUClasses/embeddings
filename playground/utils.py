import torch
import sys
import os

#hugging face login from environmental variable set in .bash_profile
from huggingface_hub import login
login(token=f"{os.environ.get('HUGGING_FACE_TOKEN')}", add_to_git_credential=True)  # ADD YOUR TOKEN HERE

# modelname='sentence-transformers/multi-qa-MiniLM-L6-dot-v1' #not normalized, suitable for dot product not cosign similarity 
modelname='sentence-transformers/multi-qa-MiniLM-L6-cos-v1'  #cosign similarity
# modelname='sentence-transformers/msmarco-MiniLM-L6-cos-v5'
# modelname='BAAI/bge-base-en-v1.5'

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