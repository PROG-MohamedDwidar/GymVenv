import gymnasium as gym
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    env.reset()
      # agent policy that uses the observation and info
    #action = env.action_space.sample()  # agent policy that uses the observation and info
    score=0
    terminated=False
    truncated=False
    while not terminated and not truncated:
        action = random.choice([0, 1])
        observation, reward, terminated, truncated, info = env.step(action)
        #print(observation, reward, terminated, truncated, info)
        score+=reward
        print(score)
        env.render()
    
    

env.close()