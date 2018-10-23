from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Dense
from keras.callbacks import TensorBoard, EarlyStopping
from keras.initializers import RandomUniform

from collections import deque
import numpy as np 
import random
from pathlib import Path

on_path = "on_model.h5"
off_path = "off_model.h5"

UPDATE_FREQUENCY = 50

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
        self.on_model = self.load_model(on_path)
        self.off_model = self.load_model(off_path)

    def load_model(self, path):
        saved_file = Path(path)

        if saved_file.is_file():
            return load_model(path)
        else:
            return self.init_model()

    def init_model(self):
        model = Sequential()
        model.add(Dense(100, kernel_initializer='VarianceScaling', input_dim=self.state_size, activation='relu'))
        model.add(Dense(60, kernel_initializer='VarianceScaling', activation='relu'))
        Q_initializer = RandomUniform(minval=-1e-6, maxval=1e-6, seed=None)
        model.add(Dense(self.action_size, kernel_initializer=Q_initializer))
        model.compile(loss='mse', optimizer=Adam(lr=self.l_rate))

        return model

    def action(self, state, step):
        if step % UPDATE_FREQUENCY == 0: reset_target()

        if np.random.rand() <= self.eps:
            return random.randrange(self.action_size)   
        
        actions = self.on_model.predict(state)
        
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
                target += self.discount * np.amax(self.off_model.predict(next_state)[0])
            
            target_f = self.on_model.predict(state)
            target_f[0][action] = target

            self.on_model.fit(state, target_f, epochs=1, verbose=0)
        
        self.on_model.save(on_path)
        self.off_model.save(off_path)

        if self.eps > self.eps_min:
            self.eps *= self.eps_decay

    def reset_target():
        self.off_model.set_weights(self.on_model.get_weights())