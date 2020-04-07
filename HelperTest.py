import Helper as helper
import HelperBase2 as h
import BoardTree2 as bt
import random

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
    ordered = np.array([[3,2,99,1],
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


def test3():
    b = np.array([[  2,   4 ,   8 ,   2],
                  [  4 ,   2 ,  16 ,   2],
                  [ 32 , 256 ,  32 ,   4],
                  [ 64 , 128 , 256 , 512]])
    board = helper.Board(b, num_levels = 5)

    rewards = board.calc_rewards()

    print(rewards)

def test4():
    b =np.array([[ 3,  4,  1,  1],
                [ 4,  6,  5, 11],
                [ 6,  7,  6,  2],
                [ 7,  8,  9,  1]])
    board = bt.BoardTree(b, num_levels = 5)

    qstates, nstates = h.randomNewStates(b)

    print(board.calc_rewards())

    '''print(board.children)

    print(board.children[3][0].reward)'''

def test5():
    board = np.array([[ 1,  2,  1,  0],
                      [ 2,  4,  2,  1],
                      [ 3,  5,  4,  2],
                      [ 5,  8,  9, 10]])

    qstates, nstates = h.randomNewStates(board)

    print(random.choice(nstates[1]))

    head = bt.BoardTree(board, num_levels = 5)

    rewards = head.calc_rewards()
    action = np.argmax(np.array(rewards))

    print(action)

test5()
