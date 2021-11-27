import sys
from collections import deque

import pylab
import random
import numpy as np
from environment import Env
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

class DeepSARSAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.state_size = state_size
        self.action_size = action_size


        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = .999
        self.epsilon_min = 0.01
        #self.model = DeepSARSA(self.state_size, self.action_size, self.lr)

        self.batch_size = 64
        self.train_start = 1000

        # 리플레이 메모리, 최대 크기 2000
        self.memory = deque(maxlen=2000)

        # 모델과 타깃 모델 생성
        self.model = self.build_model()
        self.target_model = self.build_model()

        # 타깃 모델 초기화
        self.update_target_model()

    '''
        # DQN code
        def build_model(self):
            model = Sequential()
            model.add(Dense(24, input_dim=self.state_size, activation='relu',
                            kernel_initializer='he_uniform'))
            model.add(Dense(24, activation='relu',
                            kernel_initializer='he_uniform'))
            model.add(Dense(self.action_size, activation='linear',
                            kernel_initializer='he_uniform'))
            model.summary()
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
            return model
    '''


    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Dense(30, input_dim=state_size, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:

            return random.randrange(self.action_size)
        else:

            # state = np.float32(state)
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    '''
        def train_model(self, state, action, reward, next_state, next_action, done):
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
    
    
            target_Q = self.model.predict(state)[0]
            next_Q = self.model.predict(next_state)[0][next_action]
    
    
            if done:
                target_Q[action] = reward
            else:
                target_Q[action] = (reward + self.discount_factor * next_Q )
    
    
            target_Q = np.reshape(target_Q, [1, self.action_size])
    
            self.model.fit(state, target_Q, epochs=1, verbose=0)
    '''


    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        # 현재 상태에 대한 모델의 큐함수
        # 다음 상태에 대한 타깃 모델의 큐함수
        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        # 벨만 최적 방정식을 이용한 업데이트 타깃
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        self.model.fit(states, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


if __name__ == "__main__":

    env = Env(render_speed=0.00001)
    state_size = 15
    action_space = [0, 1, 2, 3, 4]
    action_size = len(action_space)
    agent = DeepSARSAgent(state_size, action_size)

    scores, episodes = [], []

    for episode in range(201):
        done = False
        score = 0

        state = env.reset()
        state = np.reshape(state, [1, state_size])
        time = 0

        while not done:
            #add
            if agent.render:
                env.render()

            action = agent.get_action(state)


            #next_state, reward, done, info = env.step(action)
            next_state, reward, done  = env.step(action)

            next_state = np.reshape(next_state, [1, state_size])
            #next_action = agent.get_action(next_state)
            # add 에피소드가 중간에 끝나면 -100 보상
            reward = reward if not done or score == 499 else -100


            #agent.train_model(state, action, reward, next_state, next_action, done)
            #code add
            # 리플레이 메모리에 샘플 <s, a, r, s'> 저장
            agent.append_sample(state, action, reward, next_state, done)
            # 매 타임스텝마다 학습
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            score += reward
            state = next_state
            time += 1
            if time > 200:
                done = True

            if done:
                # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                agent.update_target_model()

                score = score if score == 500 else score + 100
                # 에피소드마다 학습 결과 출력
                scores.append(score)
                episodes.append(episode)
                print("episode:", episode, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)
                '''
                print("episode: {:3d} | score: {:3d} | epsilon: {:.3f}".format(
                      episode, score, agent.epsilon))
                scores.append(score)
                episodes.append(episode)
                '''

            # add 이전 10개 에피소드의 점수 평균이 490보다 크면 학습 중단
            if np.mean(scores[-min(10, len(scores)):]) > 490:
                agent.model.save_weights("./mountain_car.h5")
                sys.exit()

        if episode % 100 == 0:
            agent.model.save_weights('save_model/model', save_format='tf')
            pylab.plot(episodes, scores, 'b')
            pylab.xlabel("episode")
            pylab.ylabel("score")
            pylab.savefig("./save_graph/graph.png")

    #agent.model.save_weights('save_model/model', save_format='tf')

