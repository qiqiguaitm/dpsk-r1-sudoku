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
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a Sudoku puzzle. After thinking, when you finally complete the grid so that each row, column and 3×3 box contains the digits 1 to 9 without repeats, please clearly give the completed grid within <answer> </answer> tags. i.e., <answer>[[6, 5, 3, 4, 1, 8, 7, 2, 9], [2, ......, 6], ......, [3, 4, 2, 8, 6, 9, 5, 1, 7]]</answer>.\n<|im_end|>\n<|im_start|>user\n以下是一些基本步骤和技巧，帮助你更有效地解决数独谜题：
1. **基础排除法**：这是最基本的解数独的方法，通过检查每一行、列以及3x3的小方块（宫），确定哪些数字可以放在空格中。如果一个数字在某行、列或宫中没有出现，并且只有一个位置可以放置它，那么这个位置就应该填上该数字。
2. **唯一候选数法（隐式唯一数）**：当一个格子内的所有可能数字（候选数）被其他格子中的数字排除后，只剩下一个数字时，这个数字就是这个格子的答案。
3. **区块排除法**：如果在某个3x3的宫内，某数字的所有可能位置都集中在一行或一列中，那么这一行或一列的其它宫就不能再包含这个数字。
4. **显式/隐式数对、三链数等**：当你发现两个或三个格子在同一行、列或宫中只能填入相同的两个或三个数字时，这些数字就成为了这些格子的唯一候选数，同时也可以从同一行、列或宫的其他格子中排除这些数字。
5. **X翼、剑鱼等高级技巧**：这些是更高级的解题技巧，涉及到跨多个行或列寻找特定数字的唯一可能位置，从而进一步排除其他可能性。
6. **假设与反证**：当常规方法无法继续推进时，可以尝试做出假设（选择一个合理的候选数填充），然后继续解题。如果在这个假设下遇到了矛盾，则说明最初的假设不成立，需要回溯并选择其他候选数进行尝试。
为了更高效地应用上述技巧，建议：
- **保持清晰的笔记**：使用铅笔标记每个格子的可能数字，以便快速识别潜在的解。
- **优先处理可能性较少的格子**：首先解决那些只有少数几个候选数的格子，因为它们更容易确认。
- **定期复查整个数独盘面**：随着新数字的添加，之前不可能的情况可能会变得可行，因此定期回顾整个盘面非常重要。
请参考上诉步骤和建议，解决这个数独题目:{puzzle}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
    prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a Sudoku puzzle. After thinking, when you finally complete the grid so that each row, column and 3×3 box contains the digits 1 to 9 without repeats, please clearly give the completed grid within <answer> </answer> tags. i.e., <answer>[[6, 5, 3, 4, 1, 8, 7, 2, 9], [2, ......, 6], ......, [3, 4, 2, 8, 6, 9, 5, 1, 7]]</answer>.\n<|im_end|>\n<|im_start|>user\nSolving Sudoku puzzles can be an enjoyable and rewarding challenge. Here are some essential strategies and tricks to help you solve Sudoku puzzles more efficiently:
1. **Single Position (Naked Singles)**: If a cell has only one possible candidate, that number must go in that cell.
2. **Single Candidate (Hidden Singles)**: When a number within a row, column, or 3x3 grid appears as a candidate only once, it must be placed in that cell.
3. **Scanning**: Look for rows, columns, and boxes where only one number is missing. This often leads to filling in the missing number quickly.
4. **Crosshatching**: Focus on one 3x3 box at a time, eliminating numbers already present in intersecting rows and columns to find the remaining numbers.
5. **Subset Techniques**:
   - **Naked Pairs/Triples**: When two cells in the same row, column, or box have the same pair of candidates and no others, those candidates can be eliminated from other cells in that row, column, or box.
   - **Hidden Pairs/Triples**: Similar to naked pairs/triples, but instead of looking for cells with identical candidates, you look for cases where the pair or triple is hidden among other candidates.
6. **Pointing Pairs/Triples**: If a pair or triple in one box points to a row or column where the same numbers are the only candidates, they can be eliminated from the rest of the row or column outside the box.
7. **X-Wing**: When the same number is a candidate in exactly two cells in each of two rows (or columns), and these cells are in the same two columns (or rows), you can eliminate that number as a candidate in all other cells in those columns (or rows).
8. **Swordfish**: An extension of the X-Wing technique involving three rows or columns instead of two.
9. **XY-Wing**: Involves three cells, each containing exactly two numbers. If these cells form a pattern where one cell shares a candidate with each of the other two, you can eliminate that shared candidate from other cells sharing regions with the three involved cells.
10. **Forcing Chains**: A series of implications that link cells together, allowing you to deduce which candidates must be true based on the assumption that certain candidates are either true or false.
11. **Guessing**: As a last resort, when logical deductions fail, you might choose to make an educated guess. However, this method is less preferred because it can lead to errors if not carefully managed.
Mastering these techniques will significantly improve your ability to solve Sudoku puzzles, even at higher difficulty levels. Start by applying the simpler methods and gradually incorporate more complex strategies as needed. Happy solving! pls solve the puzzle:{puzzle}\n<|im_end|>\n<|im_start|>assistant\n<think>"""

    return prefix


def make_dataset_base(mode,local_dir,template_type,data_source):
    
    TRAIN_SIZE=10240
    TEST_SIZE=128
    N = TRAIN_SIZE+TEST_SIZE
     
    dataset = gen_dataset(N=N,mode=mode)    
    train_dataset = dataset[:TRAIN_SIZE]
    test_dataset = dataset[-TEST_SIZE:]
    dataset_dir=local_dir + '/' + 'cl_sudoku_tips_'+mode+'_'+str(TRAIN_SIZE)
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
    
        
 

   
    
    
