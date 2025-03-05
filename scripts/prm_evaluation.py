import argparse
import numpy as np
from datasets import load_dataset
from sal.config import Config

from sal.models.reward_models import load_prm
from sal.utils.parser import H4ArgumentParser


if __name__ == '__main__':
    parser = H4ArgumentParser(Config)
    configs = parser.parse()

    prm_name = configs.prm_path.split('/')[-1]
    dataset_names = ['gsm8k', 'math', 'olympiadbench', 'omnimath']

    prm = load_prm(configs)
    all_f1, all_acc_correct, all_acc_error = [], [], []
    for name in dataset_names:
        save_path = f"{configs.processbench_output_dir}/{prm_name}/{name}.jsonl"
        if not configs.evaluation:
            dataset = load_dataset('Qwen/ProcessBench', split=name)
            dataset = dataset.map(
                prm.get_prediction,
                desc=f"Running {name} with {prm_name}",
            )
            dataset.to_json(save_path, lines=True)
        else:
            dataset = load_dataset("json", data_files=save_path, split="train")

        data_error = [e for e in dataset if e['label'] != -1]
        data_correct = [e for e in dataset if e['label'] == -1]
        
        acc1 = np.mean([e['match'] for e in data_error]) * 100
        true_negative = np.sum([e['match'] for e in data_error])
        acc2 = np.mean([e['match'] for e in data_correct]) * 100
        true_positive = np.sum([e['match'] for e in data_correct])
        f1 = 2 * acc1 * acc2 / (acc1 + acc2)
        all_f1.append(f1)
        all_acc_error.append(acc1)
        all_acc_correct.append(acc2)
        print(
            f'Using {prm_name} on {name}: f1: {f1:.1f}, '
            f'error acc: {acc1:.1f} ({true_negative}/{len(data_error)}), '
            f'correct acc: {acc2:.1f} ({true_positive}/{len(data_correct)})'
        )

    all_f1 = np.mean(all_f1)
    all_acc_correct = np.mean(all_acc_correct)
    all_acc_error = np.mean(all_acc_error)
    print(
        f"Overall F1: {all_f1:.1f}; "
        f"Overall Error Acc: {all_acc_error:.1f} "
        f"Overall Correct Acc: {all_acc_correct:.1f} "
    )

