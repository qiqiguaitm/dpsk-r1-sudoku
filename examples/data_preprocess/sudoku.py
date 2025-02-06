# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import re
import os
from datasets import Dataset
import argparse


from random import randint, seed
from tqdm import tqdm
import numpy as np

from sudoku_core import board2str, gen_sudoku_dataset as gen_dataset
    


def make_prefix(dp, template_type):
    puzzle = dp['puzzle']
    if template_type == 'base':
        prefix = f"""A conversation between Uer and Assistant. The User gives a Sudoku grid, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> completed grid here </answer>. 
User: Here is a sudoku grid:\n{puzzle}and please use logic to fill in the missing digits and complete the grid so that each row, column and 3×3 box contains the digits 1 to 9 without repeats.
Assistant: Let me solve this step by step.\n<think>"""
    elif template_type == 'qwen':
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a Sudoku puzzle. After thinking, when you finally complete the grid so that each row, column and 3×3 box contains the digits 1 to 9 without repeats, please clearly give the completed grid within <answer> </answer> tags. i.e., <answer>[[6, 5, 3, 4, 1, 8, 7, 2, 9], [2, ......, 6], ......, [3, 4, 2, 8, 6, 9, 5, 1, 7]]</answer>.\n<|im_end|>\n<|im_start|>user\n{puzzle}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
        #prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>.\n<|im_end|>\n<|im_start|>user\n{quiz}\n<|im_end|>\n<|im_start|>assistant\n<think>"""     
    return prefix


def make_dataset_base(mode,local_dir,template_type,data_source):
    
    TRAIN_SIZE=102400
    TEST_SIZE=512
    N = TRAIN_SIZE+TEST_SIZE
     
    dataset = gen_dataset(N=N,mode=mode)    
    train_dataset = dataset[:TRAIN_SIZE]
    test_dataset = dataset[-TEST_SIZE:]
    dataset_dir=local_dir + '/' + 'cl_sudoku_'+mode+'_'+str(TRAIN_SIZE)
    print(dataset_dir)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            solution = example['solution']
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "game",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            
            #print(data["prompt"][0]["content"])
            
            return data

        return process_fn
    
    def to_dataset(dataset_list):
        dataset_dict = {
            "puzzle": [],
            "solution": [],
        }
        for dp in dataset_list:
            dataset_dict["puzzle"].append(dp[-2])
            dataset_dict["solution"].append(dp[-1])
        return Dataset.from_dict(dataset_dict)

    train_dataset = to_dataset(train_dataset)
    test_dataset = to_dataset(test_dataset)

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    train_dataset.to_parquet(os.path.join(dataset_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(dataset_dir, 'test.parquet'))


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/sudoku')
    parser.add_argument('--template_type', type=str, default='qwen')
    args = parser.parse_args()
    
    local_dir=args.local_dir
    template_type=args.template_type
    data_source = 'tim/sudoku'
    
    modes=['simple','medium','hard']
    for mode in modes:
        make_dataset_base(mode,local_dir,template_type,data_source)
    
        
 

   
    
    
