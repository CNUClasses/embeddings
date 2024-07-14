#see https://www.pinecone.io/learn/series/nlp/fine-tune-sentence-transformers-mnr/
from myimports import *
import utils as ut

LOGGER=None
def main():
    '''to call this scriptpython3 finetune_on_anchor_positive.py --log_fn custom_log.log --mode w
    
    '''
    global LOGGER
    parser = argparse.ArgumentParser(description="Finetune on anchor, positive pairs")
    parser.add_argument('--log_fn', type=str, default='logfile.log', help='a log filename to record results (default: logfile.log)')
    parser.add_argument('--mode', type=str, choices=['a', 'w'], default='a', help='mode to open the log file: "a" for append, "w" for write/truncate (default: "a")')  
    args = parser.parse_args()

     # Set up the LOGGER
    LOGGER = ut.setup_logger(args.log_fn, args.mode)
    startTime = time.time()

    #what model are we using
    modelname=f"{ut.modelname.split('/')[-1]}"

    # 1. Load a model to finetune with 2. (Optional) model card data
    #un-finetuned
    model = SentenceTransformer(ut.modelname,device="cuda:0" if torch.cuda.is_available() else "cpu",)

    #if already finetuned
    # model = SentenceTransformer(f"./models/{modelname}",device="cuda:0" if torch.cuda.is_available() else "cpu",)

    # 3. Load a dataset to finetune on
    train_dataset = load_dataset("json", data_files="../data/trn_with_hard_negatives.json", split="train")
    eval_dataset = load_dataset("json", data_files="../data/eval_with_hard_negatives.json", split="train")
    test_dataset = load_dataset("json", data_files="../data/tst_with_hard_negatives.json", split="train")

    #process the datasets
    # generate data for informationretreival evaluator
    # Convert the datasets to dictionaries
    corpus_dataset = concatenate_datasets([train_dataset, eval_dataset, test_dataset])

    #collect all positives from train,eval,test
    corpus = dict(
        zip(corpus_dataset["id"], corpus_dataset["positive"])
    )  # Our corpus (cid => document)

    #get queries and relevant docs
    # eval_queries,eval_relevant_docs=ut.get_queries_and_relevant_docs(eval_dataset)
    test_queries, test_relevant_docs=ut.get_queries_and_relevant_docs(test_dataset)

    # drop the id column from the datasets (otherwise they will be considered as inputs
    # want only anchor and positive columns
    train_dataset = train_dataset.remove_columns(["id"])
    eval_dataset = eval_dataset.remove_columns(["id"])
    test_dataset = test_dataset.remove_columns(["id"])

    # 4. Define a loss function
    loss = losses.MultipleNegativesRankingLoss(model)

    # 5. (Optional) Specify training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"models/{modelname}_MNRL",
        # Optional training parameters:
        num_train_epochs=4,
        per_device_train_batch_size=ut.batch_size,
        per_device_eval_batch_size=ut.batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        run_name=f"{modelname}_MNRL",  # Will be used in W&B if `wandb` is installed
    )

    # 6. Evaluaters (for eval and test datasets)
    # eval_evaluator = InformationRetrievalEvaluator(
    #     queries=eval_queries,
    #     corpus=corpus,
    #     relevant_docs=eval_relevant_docs,
    #     name=modelname,)

    test_evaluator = InformationRetrievalEvaluator(
        queries=test_queries,
        corpus=corpus,
        relevant_docs=test_relevant_docs,
        name=modelname,)

    LOGGER.info(f"--------- Base {modelname}_MNRL performance:")
    LOGGER.info(test_evaluator(model))
    LOGGER.info(f'---------')

    # 7. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        # evaluator=eval_evaluator,             #if have an evaluator it will be run on the 2000 row eval dataset every 500 steps, slows it down
    )
    trainer.train()

    LOGGER.info(f"--------- After pretraining {modelname}_MNRL performance:")
    LOGGER.info(test_evaluator(model))
    LOGGER.info(f'---------')

    ut.log_execution_time(LOGGER,startTime)

    # 8. Save the trained model
    # model.save_pretrained(f"models/{modelname}_MNRL1/final")

if __name__ == "__main__":
    main()
