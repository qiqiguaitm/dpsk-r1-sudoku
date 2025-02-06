from random import randint, seed
from tqdm import tqdm
import numpy as np
import sudokum
import pickle
import time
#https://github.com/MorvanZhou/sudoku

def board2str(board):
    # type: () -> str
    base = 3
    width, height = base, base
    size = base * base
    table = ''
    cell_length = len(str(size))
    format_int = '{0:0' + str(cell_length) + 'd}'
    for i, row in enumerate(board):
        if i == 0:
            table += ('+-' + '-' * (cell_length + 1) *
                      width) * height + '+' + '\n'
        table += (('| ' + '{} ' * width) * height + '|').format(*[format_int.format(
            x) if x != 0 else '_' * cell_length for x in row]) + '\n'
        if i == size - 1 or i % height == height - 1:
            table += ('+-' + '-' * (cell_length + 1) *
                      width) * height + '+' + '\n'
    return table


def board2str_pretty(board):
    base = 3
    side = base*base

    def expandLine(line):
        return line[0]+line[5:9].join([line[1:5]*(base-1)]*base)+line[9:13]
    line0 = expandLine("╔═══╤═══╦═══╗")
    line1 = expandLine("║ . │ . ║ . ║")
    line2 = expandLine("╟───┼───╫───╢")
    line3 = expandLine("╠═══╪═══╬═══╣")
    line4 = expandLine("╚═══╧═══╩═══╝")

    symbol = " 1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    nums = [[""]+[symbol[n] for n in row] for row in board]

    board_repr = line0 + '\n'
    for r in range(1, side+1):
        board_repr += "".join(n+s for n,
                              s in zip(nums[r-1], line1.split("."))) + '\n'
        board_repr += [line2, line3,
                       line4][(r % side == 0)+(r % base == 0)] + '\n'

    return board_repr


def gen_sudoku_dataset(N,mode='simple'):
    seed(3131415927)
    hist = set()

    lower_bound = 9.0/81.0
    upper_boud = 64.0/81.0
    delta = (upper_boud-lower_bound)/3.0
    #lower_bound -= lower_bound * 0.05
    if mode == 'simple':
        start = lower_bound
        end = lower_bound+delta
    elif mode == 'medium':
        start = lower_bound+delta
        end = lower_bound+delta*2
    elif mode == 'hard':
        start = lower_bound+delta*2
        end = lower_bound+delta*3
        
    mask_rates = np.random.uniform(start, end, size=N)
    sudokus = []
    n = 0
    t0 = time.time()
    while True:
        g = sudokum.generate(mask_rate=mask_rates[n])
        s = sudokum.solve(g, max_try=50)
        
        g_b = str(g)
        s_b = str(s[1])
        
        if not s[0]:
            continue
        elif g_b in hist:
            continue
        else:
            sudokus.append((g, s, g_b, s_b))
            hist.add(g_b)
            n += 1
            if n % 1000 == 0:
                print(n, time.time()-t0)
            if n == N:
                break
    return sudokus


if __name__ == '__main__':
    g = sudokum.generate(mask_rate=0.3)
    print(str(g))
    print(board2str(g))
    print(board2str_pretty(g))
    r,s = sudokum.solve(g)
    print(r,str(s))
    print(board2str(s))
    print(board2str_pretty(s))


    '''
    s = sudokum.solve(g, max_try=30)
    print(s[1])
    print(board2str(s[1]))
    
    N = 500000
    sudokus = gen_sudoku_dataset(N)
    
    print(sudokus[-1][-2])
    print(sudokus[-1][-1])
    
    
    with open("sudokus.pkl", "wb") as f:
        pickle.dump(sudokus, f)
    '''
