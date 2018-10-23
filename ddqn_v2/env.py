from ctypes import cdll, POINTER, c_float, c_int
from random import randint
import numpy as np
import time

lib = cdll.LoadLibrary('../libenv.so')

class Env(object):
    def __init__(self):
        self.obj = lib.Env_new()
    
    def get_state_size(self):
        return lib.GetStateSize(self.obj)

    def get_action_size(self):
        return lib.GetActionSize(self.obj)

    def reset(self):

        lib.Reset.restype = POINTER(c_float * 3)
        values = lib.Reset(self.obj).contents

        st = int(values[0])
        hp = values[1]
        dist = values[2]

        state = np.reshape([st, hp, dist], [1, 3])
        time.sleep(0.2)

        return state

    def step(self, action):

        lib.Step.argtypes = [c_int]
        lib.Step.restype = POINTER(c_float * 5)

         # values: 0 -> state_val, 1-> state_%health, 2-> distance to term, 3-> reward, 4-> is_done
        values = lib.Step(self.obj, action).contents

        st = int(values[0])
        hp = values[1]
        dist = values[2]
        next_state = np.reshape([st, hp, dist], [1, 3])
        reward = values[3]
        done_val = values[4]
        done = True if done_val == 1 else False    

        return next_state, reward, done

def test_env():
    env = Env()
    print("init state: {}".format(env.reset()))

    for _ in range(200):
        action = randint(0, 3)

        next_state, reward, done = env.step(action)
        
        print("next state square: {}".format(next_state[0, 0]))
        print("next state health: {}".format(next_state[0, 1]))
        print("reward: {}".format(reward))
        print("done: {}".format(done))

        if done:
            env.reset()

#test_env()