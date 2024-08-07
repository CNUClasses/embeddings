import subprocess
import sys
import logging
import utils as ut

# what model are we using
# modelname='sentence-transformers/msmarco-distilbert-base-v2'
# num_epochs='4'
# # num_epochs='1'
# batch_size='128'  #can get away with 256 on MNRL but Triplet is 128 only

# modelname='sentence-transformers/multi-qa-mpnet-base-cos-v1'
# num_epochs='4'
# batch_size='32'

#the following is too big for my gpu
# modelname = "intfloat/e5-mistral-7b-instruct"
# batch_size='4'

# modelname = "dunzhang/stella_en_1.5B_v5"
# batch_size='8'
# num_epochs='1'

modelname = "dunzhang/stella_en_400M_v5"
batch_size='128'
num_epochs='4'

crossencoder='cross-encoder/ms-marco-MiniLM-L-12-v2'  #finetuned does not improve performance 
# crossencoder='cross-encoder/stsb-roberta-large' #POOR PERFORMER
# crossencoder='Alibaba-NLP/gte-Qwen2-7B-instruct' #huge and crashes
# crossencoder='BAAI/bge-reranker-base'
# crossencoder='cross-encoder/nli-deberta-v3-base' #entailment, neutral stuff, need cosine similarity 


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
        ('finetuneBiEncoder.py', ['--mode', 'w', '--num_epochs',num_epochs,'--resume','n', '--modelname', modelname,'--batch_size', batch_size,'--loss','MultipleNegativesRankingLoss']),
        ('mine_hard_negatives.py', []),

        #circle loss
        # ('finetuneBiEncoder.py', [ '--mode', 'a','--num_epochs',num_epochs,'--resume','y','--modelname', modelname,'--batch_size', batch_size, '--loss','CircleLoss']),
        # ('rerank_with_chromadb.py', ['--mode', 'a', '--localmodel', 'y','--loss','CircleLoss','--modelname', modelname,'--crossencoder',crossencoder]),

        #triplet losses
        ('finetuneBiEncoder.py', [ '--mode', 'a','--num_epochs',num_epochs,'--resume','n','--modelname', modelname,'--batch_size', batch_size, '--loss','TripletLoss']),
        ('finetuneBiEncoder.py', [ '--mode', 'a','--num_epochs',num_epochs,'--resume','y','--modelname', modelname,'--batch_size', batch_size, '--loss','TripletLossOnlineSemiHNMining']),
        ('finetuneBiEncoder.py', [ '--mode', 'a','--num_epochs',num_epochs,'--resume','y','--modelname', modelname,'--batch_size', batch_size, '--loss','TripletLossOnlineHNMining']),
        
        #rerankers
        ('rerank_with_chromadb.py', ['--mode', 'a', '--localmodel', 'y','--loss','TripletLoss','--modelname', modelname,'--crossencoder',crossencoder]),
        ('rerank_with_chromadb.py', ['--mode', 'a', '--localmodel', 'y','--loss','TripletLossOnlineSemiHNMining','--modelname', modelname,'--crossencoder',crossencoder]),
        ('rerank_with_chromadb.py', ['--mode', 'a', '--localmodel', 'y','--loss','TripletLossOnlineHNMining','--modelname', modelname,'--crossencoder',crossencoder]),
        ('rerank_with_chromadb.py', ['--mode', 'a', '--localmodel', 'y','--loss','MultipleNegativesRankingLoss','--modelname', modelname,'--crossencoder',crossencoder])
    ]
    
    for script, args in scripts_with_args:
        output = run_script(script, args)
        print(f"Output of {script} with arguments {args}:\n{output}")
        print("-" * 50)  # Separator for readability

if __name__ == "__main__":
    main()