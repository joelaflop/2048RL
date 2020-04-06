import Helper2048 as helper
import numpy as np

def test1():
    board = np.array([  [4,2,2,4],
                        [0,0,0,8],
                        [0,0,8,2],
                        [2,2,1024,2048]])
    print(board)

    qstates, nstates = helper.newStates(board)

    print(qstates[3])
    print(nstates[3][0])


def test2():
    ordered = np.array([  [3,2,99,1],
                        [4,5,6,8],
                        [12,11,10,9],
                        [13,14,16,15]])
    board = np.array([  [4,3,2,1],
                        [4,5,6,8],
                        [12,11,10,16],
                        [13,14,15,9]])

    board1 = np.array([ [4,3,2,1],
                        [5,6,7,8],
                        [12,11,10,9],
                        [13,14,16,15]])

    #print(helper.zigzagOrder(ordered))

    print(helper.unsortedness(board))

    print(helper.unsortedness(board1))

def test3():
    b = np.array([[  2,   4 ,   8 ,   2],
                  [  4 ,   2 ,  16 ,   2],
                  [ 32 , 256 ,  32 ,   4],
                  [ 64 , 128 , 256 , 512]])
    board = helper.Board(b, num_levels = 5)

    rewards = board.calc_rewards()

    print(rewards)
test3()
