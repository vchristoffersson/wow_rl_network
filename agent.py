from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from collections import deque
import numpy as np 
import random


class Agent:
    def __init__(self, state_size, action_size, discount, eps, eps_decay, eps_min, l_rate):
        self.state_size = state_size       
        self.action_size = action_size
        self.mem = deque(maxlen=2000)
        self.discount = discount
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.l_rate = l_rate
        self.model = self.init_model()

    def init_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.l_rate))

        return model

    def action(self, state):
        if np.random.rand() <= self.eps:
            return random.randrange(self.action_size)   
        actions = self.model.predict(state)
        return np.argmax(actions[0])

    def remember(self, state, action, reward, next_state, terminal):
        self.mem.append((state, action, reward, next_state, terminal))

    def replay(self, batch_size):
        batch = random.sample(self.mem, batch_size)
        
        for state, action, reward, next_state, terminal in batch:
            target = reward

            if not terminal:
                target += self.discount * np.amax(self.model.predict(next_state)[0])
            
            target_f = self.model.predict(state)
            target_f[0][action] = target
            
            self.model.fit(state, target_f, epoch=1)
        
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay

