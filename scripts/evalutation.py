import json
from datasets import load_dataset
from sal.utils.math import memoized_canonical_form
from sal.utils.qwen_math_parser import extract_answer 

def compute_acc(dataset_path, pred_type):
    dataset = load_dataset("json", data_files=dataset_path)
    n_questions = dataset['train'].num_rows
    correct_counter = 0
    for x in dataset["train"]:
        pred = extract_answer(x[pred_type], "math")
        canonical_pred = memoized_canonical_form(pred)
        canonical_answer = memoized_canonical_form(x["answer"])
        if canonical_pred == canonical_answer:
            correct_counter+=1
    return correct_counter, n_questions

if __name__ == "__main__":
    # dataset_path = "/home/huidong/search-and-learn/data/meta-llama/Llama-3.2-1B-Instruct/best_of_n_completions.jsonl"
    inference_model = "Llama-3.2-1B-Instruct"
    prm_model = "Bayes-PRM"
    search_method = "beam_search" # beam_search best_of_n
    num_questions = 10
    n = 4
    dataset_path = f"data/{inference_model}/{prm_model}/" \
        f"{search_method}_n-{n}_completions-{num_questions}.jsonl"
    print(f"Inference Model: {inference_model}; PRM Model: {prm_model}; Search Method: {search_method}")

    k=2
    n_list = [2**i for i in range(1,k+1,1)]
    for n in n_list:
        pred_type = f"pred_weighted@{n}"
        n_correct, n_questions = compute_acc(dataset_path, pred_type)
        print(
            f"Prediction Type: {pred_type}; Acc: {(n_correct/n_questions*100):.1f} - ({n_correct}/{n_questions})"
        )