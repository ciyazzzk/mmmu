import torch
import os
import random
import re
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from utils.data_utils import load_yaml, construct_prompt, save_json, parse_multi_choice_response
from utils.eval_utils import eval_multi_choice, eval_open


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default='llava1.5_7b_dev.json',
                        help='name of saved json')
    parser.add_argument('--config_path', type=str, default="configs/llava1.5.yaml")
    parser.add_argument('--data_path', type=str, default="MMMU/MMMU") # hf dataset path.
    parser.add_argument('--model_name_or_path', type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument('--sub_domain', type=str, default="Art")
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    with open(args.output_path, 'r') as file:
        data = json.load(file)
    file.close()

    pred_correct = 0
    judge_dict = dict()
    metric_dict= dict()
    for sample in data:
        gold_i = sample['answer']
        pred_i = sample['parsed_pred']

        if sample['question_type'] == 'multiple-choice':
            correct = eval_multi_choice(gold_i, pred_i)
        else: # open question
            correct = eval_open(gold_i, pred_i)

        if correct:
            judge_dict[sample['id']] = 'Correct'
            pred_correct += 1
        else:
            judge_dict[sample['id']] = 'Wrong'

    if len(data) == 0:
        metric_dict.update({'acc': 0})
    metric = pred_correct / len(data)
    metric_dict.update({'acc': metric})

print(metric_dict)
