from myimports import *
import utils as ut
import chromadb
client = chromadb.Client()
from chromadb.utils import embedding_functions

# 3. Load datasets
train_dataset = load_dataset("json", data_files="../data/trn_with_hard_negatives.json", split="train")
eval_dataset = load_dataset("json", data_files="../data/eval_with_hard_negatives.json", split="train")
test_dataset = load_dataset("json", data_files="../data/tst_with_hard_negatives.json", split="train")

# generate data for informationretreival evaluator
corpus_dataset,corpus_mapper=ut.get_corpus_and_corpus_mapper(train_dataset, eval_dataset, test_dataset, dup_col='positive')

#collect all positives from train,eval,test
corpus = dict(
    zip(corpus_dataset["id"], corpus_dataset["positive"])
)  # Our corpus (cid => document)

#get uploaded fine tuned embedder
st_ef=embedding_functions.SentenceTransformerEmbeddingFunction("kperkins411/msmarco-distilbert-base-v2_triplet_legal",device='cpu')

#or from a local model
# st_ef=embedding_functions.SentenceTransformerEmbeddingFunction("./models/msmarco-distilbert-base-v2_triplet/final",device='cpu')

# Create a new chroma collection
st_collection = client.get_or_create_collection(name="st_embeddings", embedding_function=st_ef)

#add all corpus values to collection
st_collection.add(
    documents=list(corpus.values()),
    ids=[str(id) for id in list(corpus.keys())])

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

    def __init__(self, test_dataset, corpus_mapper):
        self.test_dataset = test_dataset
        self.corpus_mapper = corpus_mapper
        self.stats = defaultdict(int) #defaults to 0

    def __call__(self, i, max_score):
        """
        Updates the statistics based on the given index and maximum score.

        Parameters:
        - i (int): The index of the example.
        - max_score (int): The maximum score obtained for the example.
        """
        self.stats['totals'] += 1
        if max_score != 0:
            correct_doc = self.corpus_mapper[self.test_dataset['positive'][i]]
            original_choice = self.corpus_mapper[results['documents'][i][0]]
            reranked_choice = self.corpus_mapper[results['documents'][i][max_score]]
            if correct_doc == reranked_choice:
                self.stats['correctly_reranked'] += 1
            else:
                if correct_doc == original_choice:
                    self.stats['reranked_incorrectly'] += 1
                else:
                    self.stats['both_incorrect'] += 1

    def print_stats(self):
        """
        Prints the statistics.
        """
        print(f"Correct original predictions {self.stats['totals'] - self.stats['correctly_reranked'] - self.stats['reranked_incorrectly'] - self.stats['both_incorrect']} out of {self.stats['totals']}")
        print(f"correctly reranked {self.stats['correctly_reranked']} out of {self.stats['totals']}")
        print(f"incorrectly reranked {self.stats['reranked_incorrectly']} out of {self.stats['totals']}")
        print(f"both_incorrect {self.stats['both_incorrect']} out of {self.stats['totals']}")

#stat tracker    
ts = track_stats(test_dataset,corpus_mapper)

#this is not fine tuned!
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

#get matches for all queries
results = st_collection.query(
    # query_texts=test_dataset[0]['anchor'], #query single text
    query_texts=test_dataset['anchor'],  # Query all texts
    n_results=10    #10 results per query
)

# rerank the results with original query and documents returned from Chroma
for i in tqdm(range(len(test_dataset))):
    scores = model.predict([(test_dataset['anchor'][i], doc) for doc in results["documents"][i]])
    max_score=np.argmax(scores)
    ts(i,max_score)

ts.print_stats()
