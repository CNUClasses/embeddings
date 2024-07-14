#see https://www.pinecone.io/learn/series/nlp/fine-tune-sentence-transformers-mnr/

#this is to be run on the triplet dataset created by create_triplet_dataset_using_finetuned_model.py

from myimports import *
import utils as ut
LOGGER=None
def main():
    '''to call this script
    python3 finetune_triplet_anchor_positive_negative.py --log_fn custom_log.log --mode a
    '''

    global LOGGER
    parser = argparse.ArgumentParser(description="Finetune on anchor, positive pairs")
    parser.add_argument('--log_fn', type=str, default='logfile.log', help='a log filename to record results (default: logfile.log)')
    parser.add_argument('--mode', type=str, choices=['a', 'w'], default='a', help='mode to open the log file: "a" for append, "w" for write/truncate (default: "a")')  
    args = parser.parse_args()

     # Set up the LOGGER
    LOGGER = ut.setup_logger(args.log_fn, args.mode)
    startTime = time.time()

    # what model are we using
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
    #3a generate data for informationretreival evaluator
    # Convert the datasets to dictionaries

    corpus_dataset = concatenate_datasets([train_dataset, eval_dataset, test_dataset])
    corpus = dict(
        zip(corpus_dataset["id"], corpus_dataset["positive"])
    )  # Our corpus (cid => document)
    queries = dict(
        zip(test_dataset["id"], test_dataset["anchor"])
    )  

    #get queries and relevant docs
    # eval_queries,eval_relevant_docs=ut.get_queries_and_relevant_docs(eval_dataset)
    test_queries, test_relevant_docs=ut.get_queries_and_relevant_docs(test_dataset)

    # drop the id column from the datasets
    train_dataset = train_dataset.remove_columns(["id"])
    eval_dataset = eval_dataset.remove_columns(["id"])
    test_dataset = test_dataset.remove_columns(["id"])

    # 4. Define a loss function
    loss = TripletLoss(model=model)

    # 5. (Optional) Specify training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"models/{modelname}_triplet",
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
        run_name=f"{modelname}_triplet",  # Will be used in W&B if `wandb` is installed
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

    LOGGER.info(f'---------')
    LOGGER.info(f"Base {modelname}_triplet performance:")
    LOGGER.info(test_evaluator(model))
    LOGGER.info(f'---------')

    # 7. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        # evaluator=eval_evaluator,  #dont include saves a lot of time
    )
    trainer.train()

    LOGGER.info(f"--------- After pretraining {modelname}_triplet performance:")
    LOGGER.info(test_evaluator(model))
    LOGGER.info(f'---------')

    # 8. Save the trained model
    model.save_pretrained(f"models/{modelname}_triplet/final")

    LOGGER.info(f"--------- Script took  {datetime.now()-startTime} to run")

    # 9. (Optional) Push it to the Hugging Face Hub
    # model.push_to_hub(f"{ut.modelname.split('/')[-1]}_triplet")

    ut.log_execution_time(LOGGER,startTime)

if __name__ == "__main__":
    main()

