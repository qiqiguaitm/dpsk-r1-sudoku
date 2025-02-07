import re
from typing import Dict, Tuple, Optional
import random

def extract_question(response_str: str) -> Tuple[Optional[bool], str]:
    question_pattern = r'<\|im_start\|>user(.*?)<\|im_end\|>'
    matches = list(re.finditer(question_pattern, response_str, re.DOTALL))
    question_format_correct = False
    question_text = None
    if not matches:
        print("[Error] No valid answer tags found")    
        return question_format_correct,question_text
    else:
        question_format_correct = True
        question_text = matches[0].group(1).strip()
        return question_format_correct,question_text
    
def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
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
        print("[Error] Failed to locate model response header")
        return None, solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        print("[Error] No valid answer tags found")
        return None, processed_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str


def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
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
        
        print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
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
    do_print = random.randint(1, 8) == 1
    if do_print:  
        print("\n" + "="*80)
        print(f"Reasoning Process: {solution_str}")
        print(f"Ground Truth:\n{ground_truth}")
        print(" Processing New Sample ".center(80, '='))
        
    # Extract model answer
    answer_text, processed_str = extract_solution(solution_str)
    question_format_correct, question_text = extract_question(solution_str)
    assert question_format_correct == True
    
    if do_print:  
        print(f"\n[Model Response]\n{processed_str}")
    # Validate response structure
    format_correct = validate_response_structure(processed_str)
    format_score = format_reward if format_correct else -abs(format_reward)
    if do_print:  
        print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
        print(f"  Format score: {format_score}")


    # Validate answer content
    answer_score = 0
    if format_correct and answer_text:
        answer_text = answer_text.replace(" ","").replace("\n","")
        question_text = question_text.replace(" ","").replace("\n","")
        ground_truth = ground_truth.replace(" ","").replace("\n","")
        question_regx = re.escape(question_text).replace('0','[1-9]')
        blank_cnt = question_text.count('0')
        
        matched_len = (len(question_text) == len(answer_text)) 
        matched_keep = re.fullmatch(question_regx, answer_text)
        matched =  matched_len and matched_keep
        
        diff_cnt = sum(c1 != c2 for c1, c2 in zip(answer_text, ground_truth))
        match_ratio = 1.0-diff_cnt/blank_cnt
        diffculity = blank_cnt/64.0        
        if matched:
            if do_print:  
                print(f"\n[Content Validation]")
                print(f"  Expected : {ground_truth}")
                print(f"  Predicted: {answer_text}")
            
            if diff_cnt == 0:
                answer_score = 2.0 + 2*diffculity
                if do_print:  
                    print("  Content validation: FULL MATCH. diffculity,blank_cnt,64:",diffculity,blank_cnt,64)
            else:
                answer_score = -1.5  + 1.0 * match_ratio
                if do_print:  
                    print("  Content validation: MISMATCH; match_ratio,diff_cnt,blank_cnt:", match_ratio,diff_cnt,blank_cnt)
        else:
            answer_score = -2.0
            print( "Fail to just fill blank position, matched,matched_len,matched_keep:",matched,matched_len,matched_keep)
            
    else:
        if do_print:  
            print("\n[Content Validation] Skipped due to format errors or missing answer")


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
    response_text = "<|im_start|>user\n[[6, 4, 0, 9, 2, 7, 1, 8, 5], [8, 9, 0, 0, 4, 0, 6, 2, 7], [7, 5, 0, 6, 1, 8, 4, 3, 9], [0, 0, 4, 2, 9, 3, 5, 6, 8], [0, 2, 0, 5, 8, 4, 9, 7, 1], [0, 0, 5, 7, 6, 0, 2, 4, 0], [0, 3, 7, 4, 5, 9, 8, 1, 0], [0, 6, 8, 0, 7, 2, 3, 9, 4], [4, 1, 9, 0, 3, 6, 7, 5, 2]]\n<|im_end|>"
    format_correct,question = extract_question(response_text)
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
[[9, 7, 3, 2, 5, 4, 8, 0, 0], [5, 2, 4, 8, 0, 6, 3, 0, 0], [0, 0, 8, 7, 0, 9, 5, 2, 0], [2, 1, 7, 9, 6, 8, 4, 5, 3], [6, 4, 9, 0, 2, 5, 0, 8, 7], [8, 3, 5, 4, 0, 1, 6, 9, 2], [7, 9, 1, 5, 4, 3, 2, 6, 8], [4, 8, 6, 1, 0, 2, 7, 3, 5], [3, 5, 2, 6, 8, 7, 9, 4, 1]]
<|im_end|>
<|im_start|>assistant
<think> Utilizing Sudoku solving algorithm to fill in the empty cells ensuring all the row, column, and box constraints are met.</think>
<answer>[[9, 7, 3, 2, 5, 4, 8, 1, 6], [5, 2, 4, 8, 1, 6, 3, 7, 9], [1, 6, 8, 7, 3, 9, 5, 2, 4], [2, 1, 7, 9, 6, 8, 4, 5, 3], [6, 4, 9, 3, 2, 5, 1, 8, 7], [8, 3, 5, 4, 7, 1, 6, 9, 2], [7, 9, 1, 5, 4, 3, 2, 6, 8], [4, 8, 6, 1, 9, 2, 7, 3, 5], [3, 5, 2, 6, 8, 7, 9, 4, 1]]</answer><|im_end|>
"""
    ground_truth = f"""[[9, 7, 3, 2, 5, 4, 8, 1, 6], [5, 2, 4, 8, 1, 6, 3, 7, 9], [1, 6, 8, 7, 3, 9, 5, 2, 4], [2, 1, 7, 9, 6, 8, 4, 5, 3], [6, 4, 9, 3, 2, 5, 1, 8, 7], [8, 3, 5, 4, 7, 1, 6, 9, 2], [7, 9, 1, 5, 4, 3, 2, 6, 8], [4, 8, 6, 1, 9, 2, 7, 3, 5], [3, 5, 2, 6, 8, 7, 9, 4, 1]]"""
    score = compute_score(solution_str,ground_truth)
    print(score)