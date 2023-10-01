import gymnasium as gym
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = gym.make("LunarLander-v2",render_mode='human')

# states=env.observation_space.shape[0]
actions=env.action_space.n
# print(states)
print(actions)
model=Sequential()
# model.add(Flatten(input_shape=(1,)+states))
model.add(Flatten(input_shape=(1,)+env.observation_space.shape))
model.add(Dense(24,activation='relu'))
model.add(Dense(24,activation='relu'))
model.add(Dense(actions,activation='linear'))

agent=DQNAgent(
  model=model,
  memory=SequentialMemory(limit=50000,window_length=1),
  nb_actions=actions,
  nb_steps_warmup=10,
  target_model_update=1e-2,
  policy=BoltzmannQPolicy(),
  
  )

agent.compile(optimizer=Adam(learning_rate =0.01),metrics=['mae'])
agent.fit(env,nb_steps=100000,visualize=True,verbose=1)

results=agent.test(env,nb_episodes=100,visualize=True)
print(np.mean(results.history['episode_reward']))

env.close()
# for _ in range(1000):
#     env.reset()
#       # agent policy that uses the observation and info
#     #action = env.action_space.sample()  # agent policy that uses the observation and info
#     score=0
#     terminated=False
#     truncated=False
#     while not terminated and not truncated:
#         action = random.choice([0, 1])
#         observation, reward, terminated, truncated, info = env.step(action)
#         #print(observation, reward, terminated, truncated, info)
#         score+=reward
#         print(score)
#         env.render()
    
    



