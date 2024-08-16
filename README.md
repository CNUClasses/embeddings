# Final Recommendations for Width Project

## Summary
This report outlines key strategies and recommendations for improving the performance of semantic search and retrieval systems, particularly focusing on data preparation, model selection and training procedures for bi and cross encoders on the Hugging Face platform.

## Python files of interest
./data/convert_to_json.ipynb - notebook to convert datasets to appropriate formats
./code/runall.py - script that drives testing model/loss combinations
./code/finetuneBiEncoder.py - trains a biencoder that generates embeddings
./code/rerank_with_chromadb.py - uses biencoder above to generate embeddings, pushes them to chromadb,, then evaluates test set on chromadb.  Pulls results per query, then reranks results using reranker
./code/logs folder - contains logs of training reranking runs

## Install requirements
clone FLAG repo for hard negative mining and finetuning cross encoder
Faiss- a library for efficient similarity search and clustering of dense vectors, used by FLAG to embed contexts, 
see requirements.txt

## Data Preparation
-see ./data/???.ipynb for data prep notebook.
- Remove small contexts (1 or 2 words, typically section titles).
- Chunk the data <mark>taking care to match model context width to chunk size</mark>. Keep in mind that RAG systems use 2 models, 1 to embed and 1 to rerank.  The embedding model takes 1 sequence at a time, the reranker takes 2; a query and a context.  Ensure typical query length (in tokens) plus maximum chunk sequence length (in tokens) does not exceed the reranker models maximum sequence length.  If it does the reranker will truncate the extra tokens.

## Model Selection
-see ./code/runall.py for model test scripts
- Consult the <a href=’https://huggingface.co/spaces/mteb/leaderboard’> MTEB leaderboard< /a> for top models.  Be aware though that these scores are self reported and many models have been trained on datasets that are used to determine leaderboard position.
-Search <a href=’https://huggingface.co/models?pipeline_tag=sentence-similarity&sort=trending’>semantic_similarity models</a>for  hugging face offerings.
- Larger models generally perform better.
-Mind the default params, some use cosine similarity some use dot product.
- For RAG (Retrieval-Augmented Generation), prioritize models that excel at 'semantic similarity'.
-Smaller finetuned models are both cost effective and beat bigger general purpose LLMs (except for GPT-4), see ‘A Thorough Comparison of Cross-Encoders and LLMs for Reranking SPLADE’

## Loss Function
-see ./code/runall.py multiple loss functions were tried including 3 custom losses.
-<mark> Multiple Negative Ranking Loss (MNRL) is recommended for faster training and better performance.
- Ensure consistency in distance metrics (e.g., cosine similarity) across model training and inference.
- You must use NoDuplicatesDataLoader when using MNRL.

Loss Selection
- MNRL (Multiple Negatives Ranking Loss) on legal datasets.
- Enhance MNRL performance by building triplets offline and providing them as inputs.


## Training Biencoder
-see ./code/finetunebiencoder.py
<mark>Model performance is greatly improved by following this regime:
1. Train model A (biencoder) on anchor-positive pairs with MNRL.
2. Offline mine hard negatives (HN) using model A
3 Retrain model A with expanded dataset including HNs with MNRL.
Loop on steps 2 and 3 to get better hard negatives.</mark>

## Training Cross Encoder (reranker)
Cross-encoders serve as a second stage in RAG pipelines for reranking results. They generally provide higher accuracy than bi-encoders.  They should be fine tuned if they are going to provide improvement over biencoder selections.
- see ./code/runall.py for some of the cross encoders tested
- You must finetune the cross-encoder to get a performance boost when used with a fine tuned bi encoder
- Choose a cross-encoder at least as big as the bi-encoder.
-See < a href=”fine tuned a FLAG cross encoder. See <a href=”https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/reranker/README.md’>FLAG reranker</a> for training script 


##Mining Hard Negatives
<mark> Get many hard negatives per dataset line (15 or 20) this will expand the dataset by a factor of 15 or 20.  Use the FLAG hard negative miner, its simple, effective and already written. </mark>
-Tried 3 methods
1. custom (see ./archive/hard_negative_mining_fast.ipynb) - works OK
2. pre release Hugging Face hard negative miner.  This version had a couple of minor mistakes that I fixed. See ./code/minehardnegativesextended_function.py and mine_hard_negatives.py) – works OK,  has option of ensuring that negative is further away from positive than the anchor.  Seems like a good idea until you realize that this is a textbook hard negative.  Documentation is lean. But may of may not find any hard negatives. Works OK.
3. <a href=”https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/README.md”>FLAG hard negative mining </a> from <a href=”https://github.com/FlagOpen/FlagEmbedding/”>FLAG Embeddings</a>.  They just embed the corpus into a vector database and take the n closest points to the query.  Simple and effective.  Does not verify if any of these points are a true positive though instead of a hard negative.  Uses a model, pass it a model that is already finetuned or iterate as suggested below.  Performed well.

- Be aware that in large datasets, some hard negatives might actually answer the question, effectively making them positives. No easy way to distinguish this. 

### 3.5 Training Hints
- Use SentenceTransformers for better performance over raw PyTorch.
- Maximize batch size with MNRL for improved performance.
- For evaluation, please ensure there are no duplicate positives in the corpus (see ` ./code/utils. get_corpus_and_corpus_mapper function)
- Ensure training set is properly matched with the deduplicated corpus.


## Conclusion
Implementing these recommendations should significantly improve the performance of semantic search and retrieval systems. Continuous monitoring and fine-tuning of the process are advised for optimal results.

## Future work:
Document chunking strategy
Mine Hard Negatives- make sure HNs are not actually positives.
Train cross encoders, how to generate good training data? Maybe train using triplets and MNRL?
Eval big cross encoders CoHere, Jinja 2
Eval replacing cosign loss with CoSENTLoss and AnglELoss 