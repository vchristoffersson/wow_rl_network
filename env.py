from ctypes import cdll, POINTER, c_float, c_int
from random import randint
import numpy as np
import time

lib = cdll.LoadLibrary('libenv.so')

class Env(object):
    def __init__(self):
        self.obj = lib.Env_new()
    
    def getint(self):
        return lib.GetInt(self.obj)

    def reset(self):

        st = lib.Reset(self.obj)
        state = np.reshape([st, 1.0], [1, 2])
        time.sleep(0.5)

        return state

    def step(self, action):

        lib.Step.argtypes = [c_int]
        lib.Step.restype = POINTER(c_float * 4)

         # values: 0 -> state, 1-> %health, 2-> reward, 3-> is_done
        values = lib.Step(self.obj, action).contents

        st = int(values[0])
        hp = values[1]
        next_state = np.reshape([st, hp], [1, 2])
        reward = values[2]
        done_val = values[3]
        done = True if done_val == 1 else False    

        return next_state, reward, done

def test_env():
    env = Env()
    print("init state: {}".format(env.reset()))

    for _ in range(200):
        action = randint(0, 3)

        next_state, reward, done = env.step(action)
        
        print("next state: {}".format(next_state))
        print("reward: {}".format(reward))
        print("done: {}".format(done))

        if done:
            env.reset()

#test_env()