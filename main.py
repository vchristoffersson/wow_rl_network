import ctypes.util
from ctypes import cdll
import ctypes

from random import randint

lib = cdll.LoadLibrary('libenv.so')

class Env(object):
    def __init__(self):
        self.obj = lib.Env_new()
    
    def reset(self):
        return lib.Reset(self.obj)

    def step(self, action):

        lib.Step.argtypes = [ctypes.c_int]
        lib.Step.restype = ctypes.POINTER(ctypes.c_float * 4)

        return lib.Step(self.obj, action)

env = Env()

x = env.reset()

for i in range(200):
    action = randint(0, 3)

    ret = env.step(action).contents

    # ret: 0 -> state, 1-> %health, 2-> reward, 3-> is_done

    if ret[3] == 1:
        env.reset()