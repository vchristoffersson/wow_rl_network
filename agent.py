from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Dense
from keras.callbacks import TensorBoard, EarlyStopping

from collections import deque
import numpy as np 
import random
from pathlib import Path

path = "my_model.h5"

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
        self.model = self.load_model()

    def load_model(self):
        saved_file = Path(path)

        if saved_file.is_file():
            return load_model(path)
        else:
            return self.init_model()

    def init_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
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
        
        if len(self.mem) < batch_size:
            batch = self.mem
        else:
            batch = random.sample(self.mem, batch_size)
        
        for state, action, reward, next_state, terminal in batch:
            target = reward

            if not terminal:
                target += self.discount * np.amax(self.model.predict(next_state)[0])
            
            target_f = self.model.predict(state)
            target_f[0][action] = target
            
            #estop = EarlyStopping(monitor='val_acc', patience=10)
            #callbacks=[estop]

            self.model.fit(state, target_f, epochs=1)
        
        self.model.save(path)

        if self.eps > self.eps_min:
            self.eps *= self.eps_decay

