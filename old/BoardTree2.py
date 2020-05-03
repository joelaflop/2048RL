import numpy as np
import random
import math
import HelperBase2 as h

class BoardTree():
    def __init__ (self, board, r = 0, p = 1, move = -1, num_levels = 2):
        self.move = move
        self.p = p
        self.reward = r
        self.board = board
        self.children = []
        self.gamma = .8

        if num_levels > 0:
            qstates, nstates = h.worstNewStates(self.board) #4 lists of lists
            count = 0
            for l in nstates:
                count += len(l)
            for d, dir in enumerate(nstates):
                l = []
                for b in dir:
                    if not h.boardEquals(b, self.board):
                        l.append(BoardTree(b, r = h.manualreward(b, self.board), p = self.p/count, move = d, num_levels = num_levels-1))
                self.children.append(l)

    def calc_rewards(self):
        rewards = []
        for dir in self.children:
            r = 0
            for child in dir:
                r = max(r, child.child_rewards(1))
            rewards.append(r)
        return rewards

    def child_rewards(self, n):
        if(len(self.children) > 0):
            r = 0
            for dir in self.children:
                if dir:
                    for child in dir:
                        r = max(r, ((self.reward) + child.child_rewards(n+1))) #*(self.gamma**n)
                else:
                    r = max(r, (self.reward))
            return r if self.reward > 0 else 0
        else:
            return self.reward
