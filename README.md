# Recommendations for fine tuning RAG systems on Hugging Face bi encoders and crossencoders.

## Summary
This report outlines key strategies and recommendations for improving the performance of semantic search and retrieval systems, particularly focusing on data preparation, model selection and training procedures for bi and cross encoders on the Hugging Face platform.


## Python files of interest
./data/convert_to_json.ipynb - notebook to convert datasets to appropriate formats<br>
./code/runall.py - script that drives testing model/loss combinations<br>
./code/finetuneBiEncoder.py - trains a biencoder that generates embeddings<br>
./code/rerank_with_chromadb.py - uses biencoder above to generate embeddings, pushes them to chromadb,, then evaluates test set on chromadb.  Pulls results per query, then reranks results using reranker<br>
./code/logs folder - contains logs of training reranking runs<br>

## Install requirements
clone FLAG repo for hard negative mining and finetuning cross encoder<br>
see requirements.txt<br>

## Data Preparation
- see ./data/convert_to_json.ipynb - notebook to convert datasets to appropriate formats<br>
- Remove small contexts (1 or 2 words, typically section titles).<br>
- Chunk the data <mark>taking care to match model context width to chunk size</mark>. Keep in mind that RAG systems use 2 models, 1 to embed and 1 to rerank.  The embedding model takes 1 sequence at a time, the reranker takes 2; a query and a context.  Ensure typical query length (in tokens) plus maximum chunk sequence length (in tokens) does not exceed the reranker models maximum sequence length.  If it does the reranker will truncate the extra tokens.<br>

## Model Selection
- see ./code/runall.py - for models tested<br>
- Consult the <a href='https://huggingface.co/spaces/mteb/leaderboard'> MTEB leaderboard</a> for top models.  Be aware that these scores are self reported, many models have been trained on the same datasets that are used to determine leaderboard position.<br>
-Search <a href='https://huggingface.co/models?pipeline_tag=sentence-similarity&sort=trending'>semantic_similarity models</a>for  hugging face offerings.<br>
- Larger models generally perform better.<br>
-Mind the default params, some use cosine similarity some use dot product.<br>
-For RAG (Retrieval-Augmented Generation), prioritize models that excel at 'semantic similarity'.<br>
-Smaller finetuned models are both cost effective and beat bigger general purpose LLMs (except for GPT-4), see <a href="https://arxiv.org/html/2403.10407v1">A Thorough Comparison of Cross-Encoders and LLMs for Reranking SPLADE</a><br>

## Loss Function
- see ./code/runall.py - for loss functions tested (including 3 custom losses).<br>
- <mark> Multiple Negative Ranking Loss (MNRL) is recommended for faster training and better performance.<br>
- Ensure consistency in distance metrics (e.g., cosine similarity) across model training and inference. This means check the default distance metric for the model.<br>
- You must use NoDuplicatesDataLoader when using MNRL.</mark><br>


## Training Bi encoder
-see ./code/finetunebiencoder.py<br>
<mark>Model performance is <b>GREATLY</b> improved by following this regime:<br>
<mark> 1. Train model A (biencoder) on anchor-positive pairs with MNRL.<br>
2. Offline mine hard negatives (HN) using model A.<br>
3.  Retrain model A with expanded dataset including HNs with MNRL.<br>
<mark> Loop on steps 2 and 3 to get better hard negatives and a better model.</mark><br>

## Training Cross Encoder (reranker)
Cross-encoders serve as a second stage in RAG pipelines for reranking results. They provide higher accuracy than bi-encoders.  They must be fine tuned if they are going to be used with a fine tuned biencoder as part of a 2 stage RAG system.<br>
- see ./code/runall.py for cross encoders tested.<br>
- <mark>You must finetune the cross-encoder to get a performance boost when used with a fine tuned bi encoder.</mark><br>
- <mark>Choose a cross-encoder at least as big as the bi-encoder.<br>
- See <a href="https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/reranker/README.md">FLAG reranker</a> for training script.<br>


## Hard Negatives
<mark> The mining algorithm should search for more than 1 hard negative per dataset line.  For this project 15 hard negatives were generated per line which expanded the dataset by a factor of 15. Hard negatives can be found by training a model using MNRL loss using anchor, positive pairs.  Then use this model to embed the dataset corpus, then add the embeddings to a vector database. Finally, use the vector database to find the n closest matches to a query. </mark><br>
<mark>Be aware that in large datasets, some hard negatives might correctly answer the question, effectively making them positives. These false negatives cause the model to try to push away correct answers which degrades performence. There is no easy way to distinguish this case. The following miners suffer from this problem.</mark> 

-see ./code/HN_mining_FAISS.py<br>
See code for changing the mix of hard negatives, semi hard negatives and easy negatives. <br>

### Training Hints
- <mark>Use SentenceTransformers for better performance over raw PyTorch.
- <mark>Maximize batch size with MNRL for improved performance.
- <mark>For evaluation, ensure there are no duplicate positives in the corpus (see ./code/utils.get_corpus_and_corpus_mapper and get_queries_and_relevant_docs functions for help with this).
- <mark>Ensure training set is properly matched with the above deduplicated corpus(see ./code/utils.get_queries_and_relevant_docs functions for help with this).

## Conclusion
Implementing these recommendations will significantly improve the performance of semantic search and retrieval systems. 

## Future work:
- Effective document chunking strategies (maximum chunk size, what doc metadata to track, chunk overlap...)
- <mark>Mine Hard Negatives- how to ensure HNs are not actually positives (GPT-4o mini?)
- <mark>Train cross encoders, how to generate good training data? Maybe train using triplets and MNRL?
- Evaluate larger cross encoders (CoHere, Jinja 2).
- Evaluate replacing cosign loss with CoSENTLoss and AnglELoss 
- Full LangChain pipeline using fine tuned models for inference
- <mark>Port HN mining code 
