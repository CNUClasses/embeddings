#easy way run this BEFORE you import torch to select a particular device
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2"

#or this way, get any free GPU
from utils_gpu import get_free_gpu
get_free_gpu()

from myimports import *
from CircleLoss import CircleLoss
import utils as ut

from TripletLossOnlineHNMining import TripletLossOnlineHNMining,OnlineMineingType

LOGGER=None
def getDatasets(use_HN_dataset:str):
    """
    Returns the train, eval and test datasets based on the given loss type.

    Parameters:
    use_HN_dataset (str): Y use hard negative dataset, N use pos,anchor dataset

    Returns:
    train_dataset, eval_dataset, test_dataset: The train, eval and test datasets based on the given loss type.

    Raises:
    None
    """
    global LOGGER
    if use_HN_dataset=='N':
        LOGGER.info("Using anchor, positive dataset")
        train_dataset = load_dataset("json", data_files="../data/trn.json", split="train")
        eval_dataset = load_dataset("json", data_files="../data/eval.json", split="train")
        test_dataset = load_dataset("json", data_files="../data/tst.json", split="train")
    else:
        # LOGGER.info("Using anchor, positive, negative dataset")
        train_dataset = load_dataset("json", data_files="../data/trn_FLAG_HN.json", split="train")
        eval_dataset = load_dataset("json", data_files="../data/eval_FLAG_HN.json", split="train")

        #not as performant as the above
        # train_dataset = load_dataset("json", data_files="../data/trn_HN_KP.json", split="train")
        # eval_dataset = load_dataset("json", data_files="../data/eval_HN_KP.json", split="train")
        test_dataset = load_dataset("json", data_files="../data/tst.json", split="train")

    return train_dataset, eval_dataset, test_dataset

def getLossFunction(loss:str, batch_size:int, model):
    """
    Returns a loss function based on the given loss type.

    Parameters:
    loss (str): The type of loss function to be returned.
    batch_size: The batch size to be used in the loss function. Only relevant for CachedMultipleNegativesRankingLoss

    Returns:
    loss function: The loss function based on the given loss type.

    Raises:
    None
    """
    # 4. Define a loss function
    if loss=='MultipleNegativesRankingLoss':
        loss = losses.MultipleNegativesRankingLoss(model)  
    elif loss=='CachedMultipleNegativesRankingLoss':
        loss = CachedMultipleNegativesRankingLoss(model=model, mini_batch_size=batch_size)   
    elif loss=='CircleLoss':
        loss = CircleLoss(model=model,distance_metric=TripletDistanceMetric.COSINE)
    elif loss=='TripletLossOnlineHNMining':
        loss = TripletLossOnlineHNMining(model=model,distance_metric=TripletDistanceMetric.COSINE,triplet_margin= 0.2, OnLineMiningType=OnlineMineingType.HARDNEGATIVE)
    elif loss=='TripletLossOnlineSemiHNMining':
        loss = TripletLossOnlineHNMining(model=model,distance_metric=TripletDistanceMetric.COSINE,triplet_margin= 0.2, OnLineMiningType=OnlineMineingType.SEMIHARDNEGATIVE)   
    elif loss=='TripletLossOnlineBoth':
        loss = TripletLossOnlineHNMining(model=model,distance_metric=TripletDistanceMetric.COSINE,triplet_margin= 0.2, OnLineMiningType=OnlineMineingType.BOTH)           
    else:
        loss = TripletLoss(model=model,distance_metric=TripletDistanceMetric.COSINE, triplet_margin=.2) 
    return loss
     
def main():
    '''to call this script
    python3 finetune_triplet_anchor_positive_negative.py --num_epochs 1 --resume y --mode a --modelname sentence-transformers/multi-qa-mpnet-base-cos-v1 --batch_size 32 --loss CircleLoss
    '''

    global LOGGER
    parser = argparse.ArgumentParser(description="Finetune on anchor, positive pairs")
    # parser.add_argument('--log_fn', type=str, default='logfile.log', help='a log filename to record results (default: logfile.log)')
    parser.add_argument('--mode', type=str, choices=['a', 'w'], default='a', help='mode to open the log file: "a" for append, "w" for write/truncate (default: "a")')  
    parser.add_argument('--num_epochs', type=int, default=4, help='number epochs to finetune on (default: 4)')   
    parser.add_argument('--resume', type=str, choices=['y', 'n'], default='n', help='resume using previous models ("y") or load original pretrained model ("n") (default: "n")')  
    parser.add_argument('--modelname', type=str, default='sentence-transformers/msmarco-distilbert-base-v2', help='which model to use(default: "sentence-transformers/msmarco-distilbert-base-v2")')  
    parser.add_argument('--batch_size', type=str, default='32', help='batch size for model (default: "32")')  
    parser.add_argument('--loss', type=str, choices=['MultipleNegativesRankingLoss', 'TripletLoss', 'CircleLoss','TripletLossOnlineHNMining','TripletLossOnlineSemiHNMining','CachedMultipleNegativesRankingLoss' ],default='TripletLossOnlineSemiHNMining', help='loss function, CircleLoss and TripletLossOnlineHNMining are custom (default: "TripletLossOnlineSemiHNMining")')  
    parser.add_argument('--use_HN_dataset', type=str, choices=['Y','N' ],default='Y', help='use anchor, positive dataset or anchor,positive,negative dataset (default: "Y")')  
    parser.add_argument('--save_location', type=str, default=None, help='subdirectory where the final model is serialized. If none defaults to the name of the loss function. (default: None )')  

    argsp = parser.parse_args()

    # what model are we using
    modelname=f"{argsp.modelname.split('/')[-1]}"

    #where will the model be loaded/saved from/to?
    save_location=argsp.loss if argsp.save_location is None else argsp.save_location

     # Set up the LOGGER
    LOGGER = ut.setup_logger(modelname, argsp.mode)
    startTime = time.time()

    #set cuda device
    # torch.cuda.set_device(2)

    # 1. Load a model to finetune with 2. (Optional) model card data
    if(argsp.resume=='n'):
        #original
        ut.login_hf()   #need this because GPU server keeps going down and scripts fail
        print(f"Loading original model {argsp.modelname}")
        model = SentenceTransformer(argsp.modelname)
        # model = SentenceTransformer(argsp.modelname,trust_remote_code=True,device=f"cuda:2" if torch.cuda.is_available() else "cpu",)
        # model = SentenceTransformer(argsp.modelname,trust_remote_code=True,device=f"cuda:{ut.get_free_gpu()}" if torch.cuda.is_available() else "cpu",)
    else:
        #finetuned
        print(f"Loading finetuned model {modelname}")
        model = SentenceTransformer(f"models/{modelname}/{save_location}/final")

        # model = SentenceTransformer(f"models/{modelname}/{save_location}/final",device=f"cuda:2" if torch.cuda.is_available() else "cpu",)
        # model = SentenceTransformer(f"models/{modelname}/{save_location}/final",device=f"cuda:{ut.get_free_gpu()}" if torch.cuda.is_available() else "cpu",)

    # 3. Load a dataset to finetune on
    train_dataset, eval_dataset, test_dataset = getDatasets(argsp.use_HN_dataset)
 
    # generate data for informationretreival evaluator
    corpus_dataset,corpus_mapper=ut.get_corpus_and_corpus_mapper(train_dataset, eval_dataset, test_dataset, dup_col='positive')
 
    #collect all positives from train,eval,test
    corpus = dict(
        zip(corpus_dataset["id"], corpus_dataset["positive"])
    )  # Our corpus (cid => document)

    #get queries and relevant docs
    # eval_queries,eval_relevant_docs=ut.get_queries_and_relevant_docs(eval_dataset,corpus_mapper)
    test_queries, test_relevant_docs=ut.get_queries_and_relevant_docs(test_dataset,corpus_mapper)

    # drop the id column from the datasets
    try:
        test_dataset = test_dataset.remove_columns(["id"])
        train_dataset = train_dataset.remove_columns(["id"])
        eval_dataset = eval_dataset.remove_columns(["id"])      
    except:
        pass

    # 4. Define a loss function
    loss=getLossFunction(argsp.loss, int(argsp.batch_size), model)
    
    LOGGER.info(f"--FINETUNING -- finetunebiencoder.py; model:{modelname}, loss function:{argsp.loss}, num_epochs:{argsp.num_epochs}, batch_size:{argsp.batch_size}")

    # def compute_metrics(p):    
    #     pred, labels = p
    #     pred = np.argmax(pred, axis=1)
    #     accuracy = accuracy_score(y_true=labels, y_pred=pred)
    #     recall = recall_score(y_true=labels, y_pred=pred)
    #     precision = precision_score(y_true=labels, y_pred=pred)
    #     f1 = f1_score(y_true=labels, y_pred=pred)    
    #     return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    # 5. (Optional) Specify training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"models/{modelname}",
        # Optional training parameters:
        num_train_epochs=argsp.num_epochs,
        per_device_train_batch_size=int(argsp.batch_size),
        per_device_eval_batch_size=int(argsp.batch_size),
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES if loss=='MultipleNegativesRankingLoss' else BatchSamplers.BATCH_SAMPLER ,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        # Optional tracking/debugging parameters:
        save_total_limit = 2, # Only last 2 models are saved. Older ones are deleted.
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        run_name=f"{modelname}",  # Will be used in W&B if `wandb` is installed
        # metric_for_best_model = 'NDCG@10',
        # greater_is_better=True,
        load_best_model_at_end = True,
    )

    # 6. (Optional) Create an evaluator & evaluate the base model
    test_evaluator = InformationRetrievalEvaluator(
        queries=test_queries,
        corpus=corpus,
        relevant_docs=test_relevant_docs,
        name=modelname,)

    # dev_evaluator = TripletEvaluator(
    #     anchors=test_dataset["anchor"],
    #     positives=test_dataset["positive"],
    #     negatives=test_dataset["negative"],
    #     name=f"{modelname}",
    # )

    res=test_evaluator(model)
    ut.log_performance(res, LOGGER, modelname,loss.__class__.__name__,info="TEST SET, NOT finetuned")
 
    # 7. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        # compute_metrics=compute_metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        # evaluator=eval_evaluator,  #dont include saves a lot of time
    )
    trainer.train()

    res=test_evaluator(model)
    ut.log_performance(res, LOGGER, modelname,loss.__class__.__name__,info="TEST SET, after finetuned")
 
    # 8. Save the trained model
    model.save_pretrained(f"./models/{modelname}/{save_location}/final")

    # 9. (Optional) Push it to the Hugging Face Hub
    # model.push_to_hub(f"{ut.modelname.split('/')[-1]}_{save_location}",exist_ok=True)

    ut.log_execution_time(LOGGER,startTime)

if __name__ == "__main__":
    main()

