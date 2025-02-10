import re
from typing import Dict, Tuple, Optional
import random

def extract_question(response_str: str,do_print:bool) -> Tuple[Optional[bool], str]:
    #question_pattern = r'<\|im_start\|>user(.*?)<\|im_end\|>'
    question_pattern = r'pls solve the puzzle:(.*?)<\|im_end\|>'
    matches = list(re.finditer(question_pattern, response_str, re.DOTALL))
    question_format_correct = False
    question_text = None
    if not matches:
        if do_print:
            print("[Error] No valid answer tags found")    
        return question_format_correct,question_text
    else:
        question_format_correct = True
        question_text = matches[0].group(1).strip()
        return question_format_correct,question_text
    
def extract_solution(solution_str: str,do_print:bool) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        if do_print:
            print("[Error] Failed to locate model response header")
        return None, solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        if do_print:
            print("[Error] No valid answer tags found")
        return None, processed_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str


def validate_response_structure(processed_str: str,do_print:bool) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    if do_print:
        print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        if do_print:
            print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            if do_print:
                print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        if do_print:
            print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        if do_print:
            print("  Tag sequence validation passed")

    return validation_passed

def compute_score(solution_str: str, 
                 ground_truth: str,
                 format_reward: int = 1,
                 answer_reward: float = 1.0) :
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string
        ground_truth: Dictionary containing ground truth data
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness
        
    Returns:
        Total score (sum of format and answer rewards)
    """
    do_print = random.randint(1, 16) == 1
    do_print = 1
    if do_print:  
        print("\n" + "="*80)
        print(f"Reasoning Process: {solution_str}")
        print(f"Ground Truth:\n{ground_truth}")
        print(" Processing New Sample ".center(80, '='))
        
    # Extract model answer
    answer_text, processed_str = extract_solution(solution_str,do_print)
    question_format_correct, question_text = extract_question(solution_str,do_print)
    assert question_format_correct == True
    
    #if do_print:  
    #    print(f"\n[Model Response]\n{processed_str}")
        
        
    format_reward = 0.1
    # Validate response structure
    format_correct = validate_response_structure(processed_str,do_print)
    format_score = format_reward if format_correct else -abs(format_reward)
    if do_print:  
        print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
        print(f"  Format score: {format_score}")


    # Validate answer content
    answer_score = 0
    if answer_text:
        answer_text = answer_text.replace(" ","").replace("\n","")
        question_text = question_text.replace(" ","").replace("\n","")
        ground_truth = ground_truth.replace(" ","").replace("\n","")
        question_regx = re.escape(question_text).replace('0','[1-9]')
        blank_cnt = question_text.count('0')
        
        matched_length = (len(question_text) == len(answer_text)) 
        matched_puzzle = re.fullmatch(question_regx, answer_text)
        matched_candidate =  matched_length and matched_puzzle
        
        diff_cnt = sum(c1 != c2 for c1, c2 in zip(answer_text, ground_truth))
        match_ratio = 1.0-diff_cnt/blank_cnt
        diffculity = blank_cnt/64.0    
        if do_print:  
            print(f"\n[Content Validation]")
            print(f"  Expected : {ground_truth}")
            print(f"  Predicted: {answer_text}")    
        if matched_length:
            if matched_candidate:
                if diff_cnt == 0:
                    answer_score = 5.0 + 5.0 * diffculity
                    if do_print:  
                        print("  Content validation: GRID FULL MATCH; diffculity,blank_cnt,64:",diffculity,blank_cnt,64)
                else:
                    answer_score =  0.5 + 0.5 * match_ratio * diffculity
                    if do_print:  
                        print("  Content validation: GRID MISMATCH; match_ratio,diff_cnt,blank_cnt:", match_ratio,diff_cnt,blank_cnt)
            else:
                answer_score =  0.2
                if do_print:  
                    print( "  Content validation: PUZZLE MISMATCH; matched_candidate's,matched_length,matched_puzzle:",matched_candidate,matched_length,matched_puzzle)
        else:
            answer_score =  0.1
            if do_print:  
                print( "  Content validation: LENGTH MISMATCH; matched_candidate,matched_length,matched_puzzle:",matched_candidate,matched_length,matched_puzzle)
            
    else:
        answer_score = 0.0 
        if do_print:  
            print("  Content validation: NO ANSWER. ")
    
    '''
    1、推理格式不规范  -0.1
        1.1 答案空：-0.1
        1.2 答案长度不匹配 0.0
        1.3 填错固定位置 0.1
        1.4 答案错误 0.4-0.9
        1.5 答案正确 4.9-9.9
    2、推理格式规范 0.1
        2.1 答案空 0.1
        2.2 答案长度不匹配 0.2
        2.3 填错固定位置 0.3
        2.4 答案错误 0.6-1.1
        2.5 答案正确 5.1-10.1
        
    => 
    1、答案无效 [-0.1,0.3]; 
    2、答案错误 [0.4,1.1); 
    3、答案正确 [4.9,10.1]; 
    4、答案+格式正确 [5.1,10.1]
    '''
            
    total_score = format_score + answer_score
    if do_print:  
        print("\n" + "-"*80)
        print(f" Final Score ".center(80, '-'))
        print(f"  Format: {format_score}")
        print(f"  Answer: {answer_score}")
        print(f"  Total: {total_score}")
        print("="*80 + "\n")

    return total_score


if __name__ == '__main__':
    print("start")
    response_text = "<|im_start|>user\n pls solve the puzzle: [[6, 4, 0, 9, 2, 7, 1, 8, 5], [8, 9, 0, 0, 4, 0, 6, 2, 7], [7, 5, 0, 6, 1, 8, 4, 3, 9], [0, 0, 4, 2, 9, 3, 5, 6, 8], [0, 2, 0, 5, 8, 4, 9, 7, 1], [0, 0, 5, 7, 6, 0, 2, 4, 0], [0, 3, 7, 4, 5, 9, 8, 1, 0], [0, 6, 8, 0, 7, 2, 3, 9, 4], [4, 1, 9, 0, 3, 6, 7, 5, 2]]\n<|im_end|>"
    format_correct,question = extract_question(response_text,1)
    answer = "[[6, 4, 2, 9, 2, 7, 1, 8, 5], [8, 9, 1, 3, 4, 5, 6, 2, 7], [7, 5, 3, 6, 1, 8, 4, 3, 9], [1, 7, 4, 2, 9, 3, 5, 6, 8], [3, 2, 6, 5, 8, 4, 9, 7, 1], [9, 8, 5, 7, 6, 1, 2, 4, 3], [2, 3, 7, 4, 5, 9, 8, 1, 6], [5, 6, 8, 1, 7, 2, 3, 9, 4], [4, 1, 9, 8, 3, 6, 7, 5, 2]]"    
    answer = answer.replace(" ","")
    #answer = answer.replace("0","1")
    ground_truth="[[6, 4, 3, 9, 2, 7, 1, 8, 5], [8, 9, 1, 3, 4, 5, 6, 2, 7], [7, 5, 2, 6, 1, 8, 4, 3, 9], [1, 7, 4, 2, 9, 3, 5, 6, 8], [3, 2, 6, 5, 8, 4, 9, 7, 1], [9, 8, 5, 7, 6, 1, 2, 4, 3], [2, 3, 7, 4, 5, 9, 8, 1, 6], [5, 6, 8, 1, 7, 2, 3, 9, 4], [4, 1, 9, 8, 3, 6, 7, 5, 2]]"
    ground_truth = ground_truth.replace(" ","")
    question_regx = re.escape(question.replace(" ","")).replace('0','[1-9]')
    blank_cnt = question.count('0')
    print(blank_cnt)
    #print(re.fullmatch(question_regx, answer))
    #print(re.fullmatch(question_regx, ground_truth))
    diff_cnt = sum(c1 != c2 for c1, c2 in zip(answer, ground_truth))
    print(diff_cnt)
    print((1.0-diff_cnt/(blank_cnt+0.0))*(blank_cnt/81.0)/0.79)
    
    solution_str = f"""<|im_start|>system
You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a Sudoku puzzle. After thinking, when you finally complete the grid so that each row, column and 3×3 box contains the digits 1 to 9 without repeats, please clearly give the completed grid within <answer> </answer> tags. i.e., <answer>[[6, 5, 3, 4, 1, 8, 7, 2, 9], [2, ......, 6], ......, [3, 4, 2, 8, 6, 9, 5, 1, 7]]</answer>.
<|im_end|>
<|im_start|>user
 pls solve the puzzle: [[9, 7, 3, 2, 5, 4, 8, 0, 0], [5, 2, 4, 8, 0, 6, 3, 0, 0], [0, 0, 8, 7, 0, 9, 5, 2, 0], [2, 1, 7, 9, 6, 8, 4, 5, 3], [6, 4, 9, 0, 2, 5, 0, 8, 7], [8, 3, 5, 4, 0, 1, 6, 9, 2], [7, 9, 1, 5, 4, 3, 2, 6, 8], [4, 8, 6, 1, 0, 2, 7, 3, 5], [3, 5, 2, 6, 8, 7, 9, 4, 1]]
<|im_end|>
<|im_start|>assistant
<think> Utilizing Sudoku solving algorithm to fill in the empty cells ensuring all the row, column, and box constraints are met.</think>
<answer>[[9, 7, 3, 2, 5, 4, 8, 1, 6], [5, 2, 4, 8, 1, 6, 3, 7, 9], [1, 6, 8, 7, 3, 9, 5, 2, 4], [2, 1, 7, 9, 6, 8, 4, 5, 3], [6, 4, 9, 3, 2, 5, 1, 8, 7], [8, 3, 5, 4, 7, 1, 6, 9, 2], [7, 9, 1, 5, 4, 3, 2, 6, 8], [4, 8, 6, 1, 9, 2, 7, 3, 5], [3, 5, 2, 6, 8, 7, 9, 4, 1]]</answer><|im_end|>
"""
    ground_truth = f"""[[9, 7, 3, 2, 5, 4, 8, 1, 6], [5, 2, 4, 8, 1, 6, 3, 7, 9], [1, 6, 8, 7, 3, 9, 5, 2, 4], [2, 1, 7, 9, 6, 8, 4, 5, 3], [6, 4, 9, 3, 2, 5, 1, 8, 7], [8, 3, 5, 4, 7, 1, 6, 9, 2], [7, 9, 1, 5, 4, 3, 2, 6, 8], [4, 8, 6, 1, 9, 2, 7, 3, 5], [3, 5, 2, 6, 8, 7, 9, 4, 1]]"""
    
    
    solution_str=f''' <|im_start|>system
You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a Sudoku puzzle. After thinking, when you finally complete the grid so that each row, column and 3×3 box contains the digits 1 to 9 without repeats, please clearly give the completed grid within <answer> </answer> tags. i.e., <answer>[[6, 5, 3, 4, 1, 8, 7, 2, 9], [2, ......, 6], ......, [3, 4, 2, 8, 6, 9, 5, 1, 7]]</answer>.
<|im_end|>
<|im_start|>user
Solving Sudoku puzzles can be an enjoyable and rewarding challenge. Here are some essential strategies and tricks to help you solve Sudoku puzzles more efficiently:
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
Mastering these techniques will significantly improve your ability to solve Sudoku puzzles, even at higher difficulty levels. Start by applying the simpler methods and gradually incorporate more complex strategies as needed. Happy solving! pls solve the puzzle:[[7, 9, 3, 4, 2, 6, 1, 0, 5], [8, 1, 4, 3, 5, 9, 7, 0, 2], [6, 0, 5, 8, 1, 7, 4, 0, 9], [9, 3, 8, 6, 7, 5, 2, 1, 4], [2, 7, 6, 1, 4, 8, 9, 5, 3], [4, 5, 1, 9, 0, 2, 8, 7, 6], [3, 6, 7, 2, 8, 4, 5, 0, 1], [5, 4, 9, 7, 6, 1, 3, 0, 8], [1, 8, 2, 5, 0, 3, 0, 0, 7]]
<|im_end|>
<|im_start|>assistant
<think>

Alright, let's tackle this Sudoku puzzle step by step. First, I'll look at the given grid and identify any obvious single candidates or naked singles. I notice that in the first row, the number 0 is present, but Sudoku requires numbers from 1 to 9. So, I'll replace the 0s with the correct numbers.

Looking at the first row: [7, 9, 3, 4, 2, 6, 1, 0, 5]. The missing number is 8, so I'll place 8 in the eighth position.

Next, I'll check the second row: [8, 1, 4, 3, 5, 9, 7, 0, 2]. The missing number is 6, so I'll replace the 0 with 6.

Moving to the third row: [6, 0, 5, 8, 1, 7, 4, 0, 9]. The missing numbers are 2 and 3. Since the second position is 0, I'll determine which number fits. Looking at the column, 2 is already present in the first row, so I'll place 3 in the second position. The eighth position is also 0; since 2 is missing, I'll place 2 there.

Now, the fourth row: [9, 3, 8, 6, 7, 5, 2, 1, 4]. This row is complete, so no changes are needed.

The fifth row: [2, 7, 6, 1, 4, 8, 9, 5, 3]. This row is also complete.

The sixth row: [4, 5, 1, 9, 0, 2, 8, 7, 6]. The missing number is 3, so I'll replace the 0 with 3.

The seventh row: [3, 6, 7, 2, 8, 4, 5, 0, 1]. The missing number is 9, so I'll place 9 in the eighth position.

The eighth row: [5, 4, 9, 7, 6, 1, 3, 0, 8]. The missing number is 2, so I'll replace the 0 with 2.

Finally, the ninth row: [1, 8, 2, 5, 0, 3, 0, 0, 7]. This row has multiple 0s. The missing numbers are 4, 6, and 9. Looking at the fifth position, 4 is already present in the row, so I'll place 6 in the fifth position. The seventh position is 0; since 4 is already used, I'll place 9 there. The eighth position is also 0; the remaining number is 4, so I'll place 4 there.

After making these changes, the grid should be complete and valid, with each row, column, and 3x3 box containing all digits from 1 to 9 without repeats.
</think>

<answer>
[[7, 9, 3, 4, 2, 6, 1, 8, 5], 
 [8, 1, 4, 3, 5, 9, 7, 6, 2], 
 [6, 3, 5, 8, 1, 7, 4, 2, 9], 
 [9, 3, 8, 6, 7, 5, 2, 1, 4], 
 [2, 7, 6, 1, 4, 8, 9, 5, 3], 
 [4, 5, 1, 9, 3, 2, 8, 7, 6], 
 [3, 6, 7, 2, 8, 4, 5, 9, 1], 
 [5, 4, 9, 7, 6, 1, 3, 2, 8], 
 [1, 8, 2, 5, 9, 3, 4, 6, 7]]
</answer><｜end▁of▁sentence｜>'''
    ground_truth = f'''[[7, 9, 3, 4, 2, 6, 1, 8, 5], [8, 1, 4, 3, 5, 9, 7, 6, 2], [6, 2, 5, 8, 1, 7, 4, 3, 9], [9, 3, 8, 6, 7, 5, 2, 1, 4], [2, 7, 6, 1, 4, 8, 9, 5, 3], [4, 5, 1, 9, 3, 2, 8, 7, 6], [3, 6, 7, 2, 8, 4, 5, 9, 1], [5, 4, 9, 7, 6, 1, 3, 2, 8], [1, 8, 2, 5, 9, 3, 6, 4, 7]]'''
    score = compute_score(solution_str,ground_truth)
    print(score)