import re
import random

def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    answer_pattern = r'<answer>(.*?)</answer>'
    match_it = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match_it)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 8) == 1
    if do_print:
        print(f"--------------------------------")
        #print(f"Ground Truth:\n{ground_truth}|Extracted Answer:\n{answer}")
        print(f"Reasoning Process: {solution_str}")
        print(f"Ground Truth:\n{ground_truth}")

    if answer is None:
        if do_print:
            print(f"No answer found")
        return 0
    else:
        if answer.strip() == ground_truth.strip():
            if do_print:
                #print(f"Correct answer:\n{answer}")
                print(f"Correct answer")
            return score
        else:
            if do_print:
                #print(f"Incorrect answer:\n{answer} \n| Ground truth:\n{ground_truth}")
                print(f"Incorrect answer")
            return format_score



if __name__ == '__main__':
    s = """A conversation between Uer and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> final result here </answer>. 
User: Here is a sudoku puzzle grid:
+-------+-------+-------+
| 6   3 | 8     | 5   4 |
| 9 5 4 |   3   | 8 2 6 |
| 7   8 | 5   4 | 3 1 9 |
+-------+-------+-------+
| 8 4 6 |     2 | 9 5   |
| 5 3   |   7 6 |   8   |
| 1     | 9   8 | 4 6   |
+-------+-------+-------+
| 4 8   | 2   7 |   3 5 |
| 3   7 | 1 4 5 | 2 9 8 |
| 2   5 | 6     | 7     |
+-------+-------+-------+
and your job is to use logic to fill in the missing digits and complete the grid so that each row, column and 3×3 box contains the digits 1 to 9 without repeats.
Assistant: Let me solve this step by step.
<think> Firstly, I'll use the classic Sudoku solving strategy of scanning rows, columns, and boxes.

For example, in Column 2, we see that digits 6, 5, 7, 4, and 8 are all used, leaving digits 1, 2, 3, 9 open. Since digit 9 is already used in the same 3x3 box, it can't go into the spot with column 2. Thus, we can safely place the digit 1 there.

Similarly, I'll scan each row and box to discover missing digits, while also considering that each digit can only appear once in a row, column, or box. This solving approach will help us fill in the blank spaces efficiently and accurately. </think>

<answer>
Here is the completed Sudoku grid:

+-------+-------+-------+
| 6 3 5 | 8 1 7 | 1 4 9 |
| 9 5 4 | 1 3 2 | 8 2 6 |
| 7 1 8 | 5 6 4 | 3 1 9 |
+-------+-------+-------+
| 8 4 6 | 7 2 3 | 9 5 1 |
| 5 3 1 | 4 7 6 | 1 8 2 |
| 1 2 9 | 9 5 8 | 4 6 7 |
+-------+-------+-------+
| 4 8 2 | 3 4 7 | 6 3 5 |
| 3 6 7 | 1 4 5 | 2 9 8 |
| 2 9 5 | 6 8 1 | 7 7 3 |
+-------+-------+-------+

</answer><|endoftext|>"""
    f = extract_solution(s)
    print(f)
