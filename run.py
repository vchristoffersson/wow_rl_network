from keras.models import Sequential, load_model
from env import Env
import numpy as np 
import random

path = "my_model.h5"

model = load_model(path)

env = Env()

def pred_action(state):

    actions = model.predict(state)
    
    return int(np.argmax(actions[0]))

def run() :

    next_state = env.reset()

    is_term = False

    while not is_term:

        action = pred_action(next_state)

        next_state, reward, done = env.step(action)

        print("reward: {}".format(reward))

        is_term = done

run()