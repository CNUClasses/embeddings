import subprocess

def run_script(script_name, script_args):
    result = subprocess.run(['python3', script_name] + script_args, capture_output=True, text=True)
    return result.stdout

def main():
    scripts_with_args = [
        ('finetune_on_anchor_positive.py', ['custom_log.log', 'w']),
        ('create_triplet_dataset_using_finetuned_model.py', ['.95','.5','custom_log.log','a']),
        ('finetune_triplet_anchor_positive_negative.py', ['custom_log.log', 'w'])
    ]
    
    for script, args in scripts_with_args:
        output = run_script(script, args)
        print(f"Output of {script} with arguments {args}:\n{output}")

if __name__ == "__main__":
    main()