import subprocess
import sys
import logging
import utils as ut

# what model are we using
# modelname='sentence-transformers/msmarco-distilbert-base-v2' #max_seq_length': 350, use the cosine one below instead
# num_epochs='4'
# # num_epochs='1'
# batch_size='128'  #can get away with 256 on MNRL but Triplet is 128 only

# modelname = "intfloat/e5-mistral-7b-instruct"  #too big to train on A100
# batch_size='4'

# modelname = "dunzhang/stella_en_1.5B_v5"  #max_seq_length': 512, too big to train on A100
# batch_size='8'
# num_epochs='1'

# modelname='sentence-transformers/multi-qa-mpnet-base-cos-v1' #max_seq_length': 512
# num_epochs='2'
# batch_size='32'

# modelname = "dunzhang/stella_en_400M_v5"  #max_seq_length': 512
# batch_size='128'
# num_epochs='2'

modelname='sentence-transformers/msmarco-distilbert-cos-v5'  #max_seq_length': 384
num_epochs='2'
# num_epochs='1'
batch_size='128'  #can get away with 256 on MNRL but Triplet is 128 only
 
#rank is on https://huggingface.co/spaces/mteb/leaderboard, select the ReRanking Tab
# crossencoder='cross-encoder/ms-marco-MiniLM-L-12-v2'  #(BERT) finetuned does not improve performance 
# crossencoder='cross-encoder/stsb-roberta-large' #POOR PERFORMER
# crossencoder='Alibaba-NLP/gte-Qwen2-7B-instruct' #huge and crashes
# crossencoder='cross-encoder/nli-deberta-v3-base' #entailment, neutral stuff, need cosine similarity 
# crossencoder='intfloat/e5-large-v2'  #not a cross encoder, a biencoder, rank: 72, mem: 1.25 G :context
# crossencoder='BAAI/bge-reranker-base'  #degraded performence when reranking msmarco-distilbert-cos-v5
# crossencoder='BAAI/bge-reranker-large'  #no improvement
# crossencoder='BAAI/bge-reranker-v2-m3'
# crossencoder='models/FLAGfinetuned/checkpoint-4000-standard'
crossencoder='models/FLAGfinetuned'

def run_script(script_name, script_args):
    try:
        print(f"Starting script: {script_name} with arguments: {script_args}")
        result = subprocess.run(['python3', script_name] + script_args, 
                                capture_output=False, text=True, check=True)
        print(f"Script {script_name} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Script {script_name} failed with error code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return f"Error in {script_name}: {e.stderr}"

def main():
    scripts_with_args = [
        
        #multiple negatives ranking loss with anchor positives only
        # ('finetuneBiEncoder.py', ['--mode', 'w', '--num_epochs',num_epochs,'--resume','n', '--modelname', modelname,'--batch_size', batch_size,'--loss','MultipleNegativesRankingLoss','--use_HN_dataset','N']),

        # You need to finetune reranker otherwise it will make things worse
        
        # ('rerank_with_chromadb.py', ['--mode', 'a', '--localmodel', 'y','--loss','MultipleNegativesRankingLoss','--modelname', modelname,'--crossencoder',crossencoder]),

        #Mine hard negatives(use BAII HNM, see FLAG repo, see README.md Hard Negative Mining)
        #used finetuned on (A,P,N) multi-qa-mpnet-base-cos-v1 as the embedding model below
        # ('HN_mining.py', ['--mode', 'a', '--localmodel', 'y','--loss','MultipleNegativesRankingLoss','--modelname', 'sentence-transformers/multi-qa-mpnet-base-cos-v1','--numb_HN_per_line','15','--fraction_HN_to_semiHN','0.2']),
        ('HN_mining_FAISS.py', ['--mode', 'a', '--localmodel', 'y','--loss','MultipleNegativesRankingLoss','--modelname', 'sentence-transformers/multi-qa-mpnet-base-cos-v1','--numb_HN_per_line','15','--fraction_HN_to_semiHN','0.2']),

        #multiple negatives ranking loss with triplets
        ('finetuneBiEncoder.py', ['--mode', 'a', '--num_epochs',num_epochs,'--resume','n', '--modelname', modelname,'--batch_size', batch_size,'--loss','MultipleNegativesRankingLoss','--use_HN_dataset','Y']),
        ('rerank_with_chromadb.py', ['--mode', 'a', '--localmodel', 'y','--loss','MultipleNegativesRankingLoss','--modelname', modelname,'--crossencoder',crossencoder]),

        #CachedMultipleNegativesRankingLoss- increase batch size but slower than MultipleNegativesRankingLoss
        # ('finetuneBiEncoder.py', ['--mode', 'w', '--num_epochs',num_epochs,'--resume','n', '--modelname', modelname,'--batch_size', batch_size,'--loss','CachedMultipleNegativesRankingLoss']),
        # ('rerank_with_chromadb.py', ['--mode', 'a', '--localmodel', 'y','--loss','CachedMultipleNegativesRankingLoss','--modelname', modelname,'--crossencoder',crossencoder]),

        #triplet losses (train first with all hard negatives, then semi hard negatives, then hard negatives)
        # ('finetuneBiEncoder.py', [ '--mode', 'a','--num_epochs',num_epochs,'--resume','n','--modelname', modelname,'--batch_size', batch_size, '--loss','TripletLoss']),
        # ('rerank_with_chromadb.py', ['--mode', 'a', '--localmodel', 'y','--loss','TripletLoss','--modelname', modelname,'--crossencoder',crossencoder]),

        # ('test_GISTEmbedLoss.py', ['--mode', 'w', '--num_epochs',num_epochs,'--resume','n', '--modelname', modelname,'--batch_size', batch_size,'--loss','GISTEmbedLoss']),
        # ('rerank_with_chromadb.py', ['--mode', 'a', '--localmodel', 'y','--loss','GISTEmbedLoss','--modelname', modelname,'--crossencoder',crossencoder]),


        # #circle loss (all collapse to 1-ish cluster)
        # # ('finetuneBiEncoder.py', [ '--mode', 'a','--num_epochs',num_epochs,'--resume','y','--modelname', modelname,'--batch_size', batch_size, '--loss','CircleLoss']),
        # # ('rerank_with_chromadb.py', ['--mode', 'a', '--localmodel', 'y','--loss','CircleLoss','--modelname', modelname,'--crossencoder',crossencoder]),

        # ('finetuneBiEncoder.py', [ '--mode', 'a','--num_epochs',num_epochs,'--resume','n','--modelname', modelname,'--batch_size', batch_size, '--loss','TripletLossOnlineSemiHNMining','--save_location','TripletLoss']),
        # ('rerank_with_chromadb.py', ['--mode', 'a', '--localmodel', 'y','--loss','TripletLossOnlineSemiHNMining','--modelname', modelname,'--crossencoder',crossencoder,'--save_location','TripletLoss']),
 
        # ('finetuneBiEncoder.py', [ '--mode', 'a','--num_epochs',num_epochs,'--resume','n','--modelname', modelname,'--batch_size', batch_size, '--loss','TripletLoss','--save_location','TripletLoss']),
        # ('rerank_with_chromadb.py', ['--mode', 'a', '--localmodel', 'y','--loss','TripletLoss','--modelname', modelname,'--crossencoder',crossencoder]),

        # ('finetuneBiEncoder.py', [ '--mode', 'a','--num_epochs',num_epochs,'--resume','y','--modelname', modelname,'--batch_size', batch_size, '--loss','TripletLossOnlineHNMining','--save_location','TripletLoss']),  #mode collapse
        # ('finetuneBiEncoder.py', [ '--mode', 'a','--num_epochs',num_epochs,'--resume','n','--modelname', modelname,'--batch_size', batch_size, '--loss','TripletLossOnlineSemiHNMining','--save_location','TripletLoss']), #nope, bad perf
        # ('finetuneBiEncoder.py', [ '--mode', 'a','--num_epochs',num_epochs,'--resume','y','--modelname', modelname,'--batch_size', batch_size, '--loss','TripletLossOnlineBoth','--save_location','TripletLoss']),

        # ('rerank_with_chromadb.py', ['--mode', 'a', '--localmodel', 'y','--loss','TripletLossOnlineHNMining','--modelname', modelname,'--crossencoder',crossencoder,'--save_location','TripletLoss']),
    ]
    
    for script, args in scripts_with_args:
        output = run_script(script, args)
        print(f"Output of {script} with arguments {args}:\n{output}")
        print("-" * 50)  # Separator for readability

if __name__ == "__main__":
    main()