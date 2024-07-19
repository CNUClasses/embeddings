import subprocess
import sys
import logging
import utils as ut

#what model are we using
modelname=f"{ut.modelname.split('/')[-1]}"

# Set up the LOGGER
LOGGER = ut.setup_logger(modelname, 'w')

def run_script(script_name, script_args):
    try:
        logging.info(f"Starting script: {script_name} with arguments: {script_args}")
        result = subprocess.run(['python3', script_name] + script_args, 
                                capture_output=True, text=True, check=True)
        logging.info(f"Script {script_name} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Script {script_name} failed with error code {e.returncode}")
        logging.error(f"Error output: {e.stderr}")
        return f"Error in {script_name}: {e.stderr}"

def main():
    scripts_with_args = [
        ('finetune_on_anchor_positive.py', ['--mode', 'a', '--num_epochs','4','--resume','y']),
        ('create_triplet_dataset_using_finetuned_model.py', ['--high', '.75','--low', '.6','--mode', 'a']),
        ('finetune_triplet_anchor_positive_negative.py', [ '--mode', 'a','--num_epochs','10','--resume','y'])
    ]
    
    for script, args in scripts_with_args:
        output = run_script(script, args)
        print(f"Output of {script} with arguments {args}:\n{output}")
        print("-" * 50)  # Separator for readability

if __name__ == "__main__":
    main()