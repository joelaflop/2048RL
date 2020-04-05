import Helper2048 as helper
import numpy as np

def test():
    board = np.array([  [4,2,2,4],
                        [0,0,0,8],
                        [0,0,8,2],
                        [2,2,1024,2048]])
    print(board)

    qstates, nstates = helper.newStates(board)

    print(qstates[3])
    print(nstates[3][0])

test()
