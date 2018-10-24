from ctypes import cdll, POINTER, c_float, c_int
from random import randint
import numpy as np
import time

lib = cdll.LoadLibrary('../libenvs.so')

class Env(object):
    def __init__(self):
        self.obj = lib.Env_new()
    
    def get_state_size(self):
        return lib.GetStateSize(self.obj)

    def get_action_size(self):
        return lib.GetActionSize(self.obj)

    def reset(self):

        lib.Reset.restype = POINTER(c_float * 5)
        values = lib.Reset(self.obj).contents

        x = values[0]
        y = values[1]
        x_diff = values[2]
        y_diff = values[3]
        hp = values[4]
        state = np.reshape([x, y, x_diff, y_diff, hp], [1, 5])

        return state

    def step(self, action):

        lib.Step.argtypes = [c_int]
        lib.Step.restype = POINTER(c_float * 7)

        values = lib.Step(self.obj, action).contents

        x = values[0]
        y = values[1]        
        x_diff = values[2]
        y_diff = values[3]        
        hp = values[4]
        state = np.reshape([x, y, x_diff, y_diff, hp], [1, 5])
        reward = values[5]
        done_val = values[6]
        done = True if done_val == 1 else False    

        return state, reward, done

def test_env():
    env = Env()

    print("init state: {}".format(env.reset()))

    for _ in range(200):
        action = randint(0, 3)

        state, reward, done = env.step(action)
        
        print(state)

        print("x: {}".format(state[0, 0]))
        print("y: {}".format(state[0, 1]))
        print("x diff to avg goal: {}".format(state[0, 2]))
        print("x diff to avg goal: {}".format(state[0, 3]))
        print("next state health: {}".format(state[0, 4]))
        print("reward: {}".format(reward))
        print("done: {}".format(done))

        if done:
            env.reset()

#test_env()