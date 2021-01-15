import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque


class AI_Trader:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size  # normalized previous days
        self.action_size = 3  # sit, buy, sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.model = load_model(
             "models/" + model_name) if is_eval else self._model()

    def _model(self):
        model = Sequential()
        model.add(LSTM(units=32, return_sequences=True, activation="relu", input_shape=(7,2)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=32, activation="relu", return_sequences=True))
        model.add(Dropout(0.2))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))

        return model

    def act(self, state):    
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        options = self.model.predict(state)
        return np.argmax(options[0][0])

    def expReplay(self, batch_size):
       	mini_batch = []
        l = len(self.memory) 
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])
         
        for state, action, reward, next_state, done in mini_batch:
            reward = reward
            action = action
            if not done:
                #if agent is not in a terminal state we calculate the discounted total reward
                reward = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
                
            target = self.model.predict(state)
            target[0][0][action] = reward
            self.model.fit(state, target, epochs=1, verbose=0)
        
            history = self.model.fit(state, target_f, epochs=1, verbose=0)
            #print(history.history)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 

        return history.history