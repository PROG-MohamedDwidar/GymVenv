import math
from typing import Optional, Union
import numpy as np
import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.environments import suite_gym
from tf_agents.trajectories import trajectory
class DQNAgent:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.q_net = q_network.QNetwork(
            self.env.observation_space,
            self.env.action_space.n,
            fc_layer_params=(100,)
        )
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
        self.agent = dqn_agent.DqnAgent(
            self.env.observation_space,
            self.env.action_space,
            q_network=self.q_net,
            optimizer=self.optimizer,
            td_errors_loss_fn=tf.compat.v1.losses.huber_loss,
            train_step_counter=tf.Variable(0)
        )
        self.agent.initialize()
    
    def train(self, num_iterations):
        for i in range(num_iterations):
            time_step = self.env.reset()
            while True:
                action_step = self.agent.collect_policy.action(time_step)
                next_time_step = self.env.step(action_step.action.numpy()[0])
                traj = trajectory.from_transition(time_step, action_step, next_time_step)
                self.agent.train(experience=traj)
                time_step = next_time_step
                if time_step.is_last():
                    break
    
    def get_action(self, observation):
        time_step = self.env.reset()
        time_step = time_step._replace(observation=observation)
        action_step = self.agent.policy.action(time_step)
        return action_step.action.numpy()[0]

agent = DQNAgent('CartPole-v1')
agent.train(100)

myenv= gym.make('CartPole-v1', render_mode='human')

done=False
obs=myenv.reset()
while not done:
    action = agent.get_action(obs)
    obs, reward, done, info = myenv.step(action)
    myenv.render()
