import copy
import pylab
import random
import numpy as np
from environment import Env
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

class DeepSARSA(Sequential):
    def __init__(self, state_size, action_size, learning_rate):
        super().__init__()
        self.add(Dense(30, input_dim=state_size, activation='relu'))
        self.add(Dense(30, activation='relu'))
        self.add(Dense(action_size, activation='linear'))
        self.summary()
        self.compile(loss='mse', optimizer=Adam(lr=learning_rate))

# κ·Έλ¦¬?μ???μ ?μ???₯μ΄???μ΄?νΈ
class DeepSARSAgent:
    def __init__(self, state_size, action_size):
        # ?ν???¬κΈ°? ?λ???¬κΈ° ?μ
        self.state_size = state_size
        self.action_size = action_size

        # ?₯μ΄???μ΄???λΌλ©ν°
        self.lr=0.001
        self.epsilon = 0.01
        self.model = DeepSARSA(self.state_size, self.action_size, self.lr)
        self.model.load_weights('save_model/model')

    # ?μ€λ‘??μ ?μ±?Όλ‘ ?λ ? ν
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # λ¬΄μ?ν??λ°ν
            return random.randrange(self.action_size)
        else:
            # λͺ¨λΈλ‘λ????λ ?°μΆ
            # state = np.float32(state)
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

if __name__ == "__main__":
    # ?κ²½κ³??μ΄?νΈ ?μ±
    env = Env(render_speed=0.1)
    state_size = 15
    action_space = [0, 1, 2, 3, 4]
    action_size = len(action_space)
    agent = DeepSARSAgent(state_size, action_size)

    scores, episodes = [], []

    for episode in range(10):
        done = False
        score = 0
        # env μ΄κΈ°??
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        time = 0

        while not done:
            # ?μ¬ ?ν??????λ ? ν
            action = agent.get_action(state)

            # ? ν???λ?Όλ‘ ?κ²½?μ ????μ€??μ§ν ???ν ?μ§
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            score += reward
            state = next_state
            time += 1
            if time > 200:
                done = True

            if done:
                # ?νΌ?λλ§λ€ ?μ΅ κ²°κ³Ό μΆλ ₯
                print("episode: {:3d} | score: {:3d} | epsilon: {:.3f}".format(
                      episode, score, agent.epsilon))

