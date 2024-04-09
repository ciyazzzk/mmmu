import torch
import os
import random
import re
import numpy as np
from tqdm import tqdm

from argparse import ArgumentParser
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from utils.data_utils import load_yaml, construct_prompt, save_json, get_multi_choice_score


from transformers import LlavaForConditionalGeneration, AutoProcessor


def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def collate_fn(batches):
    return batches

def main():
    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default='llava1.5_7b_dev.json',
                        help='name of saved json')
    parser.add_argument('--config_path', type=str, default="configs/llava1.5.yaml")
    parser.add_argument('--data_path', type=str, default="MMMU/MMMU") # hf dataset path.
    parser.add_argument('--model_name_or_path', type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument('--sub_domain', type=str, default="Math")
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    
    set_seed(args.seed)

    print('llava_initializing...')
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_name_or_path
    )

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()
    # load config and process to one value
    args.config = load_yaml(args.config_path)
    for key, value in args.config.items():
        if key != 'eval_params' and type(value) == list:
            assert len(value) == 1, 'key {} has more than one value'.format(key)
            args.config[key] = value[0]


    sub_dataset = load_dataset(args.data_path, args.sub_domain, split=args.split)
    # merge all dataset
    # dataset = concatenate_datasets(sub_dataset_list)
    dataset = sub_dataset

    dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=1,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

    out_samples=[]
    for batch in tqdm(dataloader):
        for i, sample in enumerate(batch):
            if i==2:
                break
            if sample['image_2']!=None:
                continue 

            image = sample['image_1'].convert('RGB')
            if sample['question_type'] == 'multiple-choice':
                model_answer = get_multi_choice_score(sample,model,processor,args,device)
            else:
                sample = construct_prompt(sample,args.config)
                text = sample['final_input_prompt']

                inputs = processor(text=text, images=image, return_tensors='pt').to(device)
                res = model.generate(**inputs, max_new_tokens=50)
                model_answer = processor.batch_decode(res, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                start = model_answer.find('ASSISTANT:')+len('ASSISTANT:')
                model_answer = model_answer[start:]
                end = model_answer.find('.')
                model_answer = model_answer[:end]            
            
            out_result=dict(id=sample['id'],question_type=sample['question_type'],answer=sample['answer'],parsed_pred=model_answer)
            # out_samples[sample['id']] = model_answer
            # out_samples[]
            out_samples.append(out_result)

    save_json(args.output_path, out_samples)


if __name__ == '__main__':
    main()

