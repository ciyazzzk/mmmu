"""Utils for data load, save, and process (e.g., prompt construction)"""

import os
import json
import yaml
import re
import random
random.seed(42)
import numpy as np

DOMAIN_CAT2SUB_CAT = {
  'Art and Design': ['Art', 'Art_Theory', 'Design', 'Music'],
  'Business': ['Accounting', 'Economics', 'Finance', 'Manage','Marketing'],
  'Science': ['Biology', 'Chemistry', 'Geography', 'Math', 'Physics',],
  'Health and Medicine': ['Basic_Medical_Science', 'Clinical_Medicine', 'Diagnostics_and_Laboratory_Medicine', 'Pharmacy', 'Public_Health'],
  'Humanities and Social Science': ['History', 'Literature', 'Sociology', 'Psychology'],
  'Tech and Engineering': ['Agriculture', 'Architecture_and_Engineering', 'Computer_Science', 'Electronics', 'Energy_and_Power', 'Materials', 'Mechanical_Engineering'],
}


CAT_SHORT2LONG = {
    'acc': 'Accounting',
    'agri': 'Agriculture',
    'arch': 'Architecture_and_Engineering',
    'art': 'Art',
    'art_theory': 'Art_Theory',
    'bas_med': 'Basic_Medical_Science',
    'bio': 'Biology',
    'chem': 'Chemistry',
    'cli_med': 'Clinical_Medicine',
    'cs': 'Computer_Science',
    'design': 'Design',
    'diag_med': 'Diagnostics_and_Laboratory_Medicine',
    'econ': 'Economics',
    'elec': 'Electronics',
    'ep': 'Energy_and_Power',
    'fin': 'Finance',
    'geo': 'Geography',
    'his': 'History',
    'liter': 'Literature',
    'manage': 'Manage',
    'mark': 'Marketing',
    'mate': 'Materials',
    'math': 'Math',
    'mech': 'Mechanical_Engineering',
    'music': 'Music',
    'phar': 'Pharmacy',
    'phys': 'Physics',
    'psy': 'Psychology',
    'pub_health': 'Public_Health',
    'socio': 'Sociology'
}

# DATA SAVING
def save_json(filename, ds):
    with open(filename, 'w') as f:
        json.dump(ds, f, indent=4)


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    """
    
    start_chr = 'A'
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices

def load_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_dict


def parse_img_path(text):
    matches = re.findall("<img='(.*?)'>", text)
    return matches

def process_single_sample(data):
    question = data['question']
    o_imgs_paths = []
    for option in data['options']:
        current_o_imgs_paths = parse_img_path(option)
        for img_path in current_o_imgs_paths:
            o_imgs_paths.append(img_path)

    if len(o_imgs_paths) > 1:  # multiple images in options, used for random selection
        return {'id': data['id'], 'question': question, 'options': data['options'], 'answer': data['answer'],
             'image': None, 'question_type': data['question_type']}
    else:
        return {'id': data['id'], 'question': question, 'options': data['options'], 'answer': data['answer'],
             'image': data['image_1'], 'question_type': data['question_type']}


# DATA SAVING
def save_json(filename, ds):
    with open(filename, 'w') as f:
        json.dump(ds, f, indent=4)

def save_jsonl(filename, data):
    """
    Save a dictionary of data to a JSON Lines file with the filename as key and caption as value.

    Args:
        filename (str): The path to the file where the data should be saved.
        data (dict): The dictionary containing the data to save where key is the image path and value is the caption.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for img_path, caption in data.items():
            # Extract the base filename without the extension
            base_filename = os.path.basename(img_path)
            # Create a JSON object with the filename as the key and caption as the value
            json_record = json.dumps({base_filename: caption}, ensure_ascii=False)
            # Write the JSON object to the file, one per line
            f.write(json_record + '\n')

def save_args(args, path_dir):
    argsDict = args.__dict__
    with open(path_dir + 'setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')



# DATA PROCESSING
# def construct_prompt(sample, config):
#     question = sample['question']
#     options = eval(sample['options'])
#     example = ""
#     if sample['question_type'] == 'multiple-choice':
#         start_chr = 'A'
#         prediction_range = []
#         index2ans = {}
#         for option in options:
#             prediction_range.append(start_chr)
#             example += f"({start_chr}) {option}\n"
#             index2ans[start_chr] = option
#             start_chr = chr(ord(start_chr) + 1)
#         empty_prompt_sample_structure = config['multi_choice_example_format']
#         empty_prompt = empty_prompt_sample_structure.format(question, example)
#         res_dict = {}
#         res_dict['index2ans'] = index2ans
#         res_dict['correct_choice'] = sample['answer']
#         res_dict['all_choices'] = prediction_range
#         res_dict['empty_prompt'] = empty_prompt
#         if config['task_instructions']:
#             res_dict['final_input_prompt'] = config['task_instructions'].strip() + '\n\n' + empty_prompt
#         else:
#             res_dict['final_input_prompt'] = empty_prompt

#         res_dict['gt_content'] = options[ord(sample['answer'].upper()) - ord('A')]
#     else:
#         empty_prompt_sample_structure = config['short_ans_example_format']
#         empty_prompt = empty_prompt_sample_structure.format(question)
#         res_dict = {}
#         res_dict['empty_prompt'] = empty_prompt
#         if config['task_instructions']:
#             res_dict['final_input_prompt'] = config['task_instructions'].strip() + '\n\n' + empty_prompt
#         else:
#             res_dict['final_input_prompt'] = empty_prompt
#         res_dict['gt_content'] = sample['answer']

#     res_dict.update(sample)
#     return res_dict

def construct_prompt(sample, config):
    question = sample['question']
    options = eval(sample['options'])
    example = ""
    question = re.sub('<image 1>','in this image ', question)
    if sample['question_type'] == 'multiple-choice':
        start_chr = 'A'
        prediction_range = []
        index2ans = {}
        for option in options:
            prediction_range.append(start_chr)
            example += f"({start_chr}) {option}\n"
            index2ans[start_chr] = option
            start_chr = chr(ord(start_chr) + 1)
        empty_prompt_sample_structure = config['multi_choice_example_format']
        empty_prompt = empty_prompt_sample_structure.format(question, example)
        res_dict = {}
        res_dict['index2ans'] = index2ans
        res_dict['correct_choice'] = sample['answer']
        res_dict['all_choices'] = prediction_range
        res_dict['empty_prompt'] = empty_prompt
        if config['task_instructions']:
            res_dict['final_input_prompt'] = config['task_instructions'].strip() + '\n\n' + empty_prompt
        else:
            res_dict['final_input_prompt'] = empty_prompt

        res_dict['gt_content'] = options[ord(sample['answer'].upper()) - ord('A')]
    else:
        empty_prompt_sample_structure = config['short_ans_example_format']
        empty_prompt = empty_prompt_sample_structure.format(question)
        res_dict = {}
        res_dict['empty_prompt'] = empty_prompt
        if config['task_instructions']:
            res_dict['final_input_prompt'] = config['task_instructions'].strip() + '\n\n' + empty_prompt
        else:
            res_dict['final_input_prompt'] = empty_prompt
        res_dict['gt_content'] = sample['answer']

    res_dict.update(sample)
    return res_dict
    

def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " " # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f'({choice})' in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices: # e.g., A B C D
            if f' {choice} ' in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack: 
                for can in candidates:
                    index = response.rfind(f'({can})')
                    start_indexes.append(index) # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else: # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index

def get_multi_choice_score(sample,model,processor,args,device):
    """Computes the scores for each image_option / caption_option pair in the joint loader.

    Args:
        joint_loader (DataLoader): batches have "image_options" and "caption_options" fields.
        "image_options" is a list of images, and "caption_options" is a list of captions.

    Returns:
        all_scores: A numpy array containing the scores of the shape NxKxL,
        where N is the number of test cases, K is the number of image options per the test case,
        and L is the number of caption options per the test case.
    """
    im_scores = []
    image = sample['image_1'].convert('RGB')
    question = sample['question']
    for c_option in eval(sample['options']):
        # inputs["input_ids_masked"] = inputs["input_ids"].detach().clone() 
        # inputs["bool_masked_pos"] = torch.zeros_like(inputs["bool_masked_pos"])
        empty_prompt_sample_structure = args.config['multi_choice_example_format']
        prompt = empty_prompt_sample_structure.format(question, c_option)

        # qst=[prompt] * len(list(c_option))
        # concatenated_list = [s1 + s2 for s1, s2 in zip(qst, c_option)]
        
        # inputs_prefix = self.processor(text=qst, images=list(i_option),padding="max_length", return_tensors="pt", max_length=77).to(self.device)
        inputs = processor(text=prompt, images=image, padding="max_length",return_tensors="pt", max_length=60).to(device)  #torch.Size([16, 77])
#                     # prefix_length=self.processor(text=prompt, images=list(i_option)[0],return_tensors="pt").to(self.device)['input_ids'].size()[1]
                    
                        
    # # continue_ids=processor.tokenizer.encode(text=c_option,return_tensors="pt")[0][1:].to(device)
        continue_ids=processor.tokenizer.encode(text=c_option,return_tensors="pt").to(device)[0][1:]
                        
        ans_length=processor.tokenizer.encode(text=c_option,return_tensors="pt").to(device).size()[1]-1
                                        
        outputs = model(**inputs)
        outputs = outputs['logits'][0, -ans_length-1: -1, :]

        log_probs = outputs[range(outputs.shape[0]),continue_ids].sum().item()        


        log_probs=np.expand_dims(np.array(log_probs),-1)
        im_scores.append(np.expand_dims(log_probs, -1))
                # np.concatenate(im_scores, axis=-1) (16,1)
                # pdb.set_trace()

    best_index = max(range(len(im_scores)), key=lambda i: im_scores[i])      

    pred_answer = chr(65+best_index)

    return pred_answer
