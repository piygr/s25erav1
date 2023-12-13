import time

import os
import torch
from kivy.app import App
from kivy.config import Config
from kivy.core.window import Window
from kivy.clock import Clock

import numpy as np
import random

from model import TD3, ReplayBuffer, evaluate_policy

Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '2788')
Config.set('graphics', 'height', '1304')
from env import MapEnv

class CarApp(App):

    Window.size = ( int(Config.get('graphics', 'width')) / 2, int(Config.get('graphics', 'height')) / 2 )  # (1394, 652)

    def __init__(self, seed=0, policy=None, mode='eval'):
        super().__init__()

        self.mode = mode
        self.seed = random.randint(0, 10000)

        if self.mode == 'train':
            self.training_config = dict(
                start_timesteps=5e3,
                eval_freq=5e3,
                max_timesteps=5e5,
                save_models=True,
                expl_noise = 0.1,
                batch_size = 100,
                discount = 0.99,
                tau = 0.005,
                policy_noise = 0.2,
                noise_clip = 0.5,
                policy_freq = 2,
                env_name = 'MapEnv',
                max_episode_steps = 1000
            )

            #state_dim = env.observation_space.shape[0]
            #action_dim = env.action_space.shape[0]
            #max_action = float(env.action_space.high[0])


    def build(self):

        self.env = MapEnv()
        self.obs = self.env.reset(seed=self.seed)

        if self.mode == 'train':
            self.file_name = "%s_%s_%s" % ("TD3", self.training_config['env_name'], str(self.seed))
            self.policy = TD3(self.env.observation_space.shape[0], self.env.action_space.shape[0], self.env.max_action)
            self.replay_buffer = ReplayBuffer()
            self.evaluations = [evaluate_policy(self.env, self.policy)]

            self.total_timesteps = 0
            self.timesteps_since_eval = 0
            self.episode_num = 0
            self.done = True
            self.episode_timesteps = 0
            self.episode_reward = 0.0


        def render(dt):

            if self.mode == 'train':
                print("---Total Timesteps: {} ".format(
                    self.total_timesteps
                ))

                if self.done:

                    print("Total Timesteps: {} Episode Num: {} Reward: {}".format(
                        self.total_timesteps,
                        self.episode_num,
                        self.episode_reward
                    ))

                    # If we are not at the very beginning, we start the training process of the model
                    if self.total_timesteps > self.training_config['start_timesteps']:

                        self.policy.train(
                            self.replay_buffer,
                            self.episode_timesteps,
                            self.training_config['batch_size'],
                            self.training_config['discount'],
                            self.training_config['tau'],
                            self.training_config['policy_noise'],
                            self.training_config['noise_clip'],
                            self.training_config['policy_freq']
                        )

                    # We evaluate the episode and we save the policy
                    if self.timesteps_since_eval >= self.training_config['eval_freq']:
                        self.timesteps_since_eval %= self.training_config['eval_freq']
                        self.evaluations.append(evaluate_policy(self.env, self.policy))
                        self.policy.save(self.file_name, directory="./pytorch_models")
                        np.save("./results/%s" % (self.file_name), self.evaluations)

                    if (self.total_timesteps < self.training_config['start_timesteps'] and \
                    self.episode_timesteps >= self.training_config['max_episode_steps']/2) or \
                        (self.total_timesteps > self.training_config['start_timesteps'] and \
                         self.episode_timesteps >= self.training_config['max_episode_steps']):

                        self.obs = self.env.reset(seed=self.seed)

                    # Set the Done to False
                    self.done = False

                    # Set rewards and episode timesteps to zero
                    self.episode_reward = 0
                    self.episode_timesteps = 0
                    self.episode_num += 1

                # Before 10000 timesteps, we play random actions
                if self.total_timesteps < self.training_config['start_timesteps']:
                    action = self.env.action_space.sample()
                    print(self.total_timesteps, '. Sample Action: ', action)
                else:  # After 10000 timesteps, we switch to the model
                    action = self.policy.select_action(np.array(self.obs))
                    # If the explore_noise parameter is not 0, we add noise to the action and we clip it
                    if self.training_config['expl_noise'] != 0:
                        action = (action + np.random.normal(0, self.training_config['expl_noise'], size=self.env.action_space.shape[0])).clip(
                            self.env.action_space.low, self.env.action_space.high)

                    print(self.total_timesteps, '. Policy Action: ', action)
                # The agent performs the action in the environment, then reaches the next state and receives the reward
                new_obs, reward, self.done, _ = self.env.step(action)

                # We check if the episode is done
                done_bool = 0 if self.episode_timesteps + 1 == self.training_config['max_episode_steps'] else float(self.done)

                print(self.done, '------Done------', done_bool)
                # We increase the total reward
                self.episode_reward += reward

                # We store the new transition into the Experience Replay memory (ReplayBuffer)
                self.replay_buffer.add((self.obs, new_obs, action, reward, done_bool))

                # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
                self.obs = new_obs
                self.episode_timesteps += 1
                self.total_timesteps += 1
                self.timesteps_since_eval += 1

                if self.total_timesteps >= self.training_config['max_timesteps']:
                    # We add the last policy evaluation to our list of evaluations and we save our model
                    self.evaluations.append(evaluate_policy(self.env, self.policy))

                    if self.training_config['save_models']:
                        self.policy.save("%s" % (self.file_name), directory="./pytorch_models")
                        np.save("./results/%s" % (self.file_name), self.evaluations)

                    self.clk.cancel()

            elif self.mode == 'eval':
                pass
            else:
                # Take a random action
                action = self.env.action_space.sample()

                obs, reward, done, info = self.env.step(action)

                # Render the game

                if done == True:
                    self.env.reset(seed=self.seed)


        self.clk = Clock.schedule_interval(render, 1.0/60.0 )


        return self.env.canvas




# Running the whole thing
if __name__ == '__main__':
    app = CarApp(mode='train')
    app.run()