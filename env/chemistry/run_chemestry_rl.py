'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2022-07-26 21:17:43
LastEditTime: 2022-12-28 12:39:19
Description: 
'''

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from chemistry_env_rl import ColorChangingRL


num_steps = 10
movement = 'Static' # Dynamic, Static
num_objects = 6
num_colors = 2
graph = 'jungle' + str(num_objects)

env = ColorChangingRL(
    test_mode='OOD-S', 
    render_type='shapes', 
    num_objects=num_objects, 
    num_colors=num_colors, 
    movement=movement, 
    max_steps=num_steps
)

# set graph type
env.set_graph(graph)

# save the graph structure and MLP parameters
env_data = env.get_save_information()
env.load_save_information(env_data)

e_i = 1
episode_count = 500
action_space = env.action_space
for _ in tqdm(range(episode_count), leave=False):
    state = env.reset(num_steps=num_steps, stage='test')
    rewards = []
    for i in range(num_steps):
        action = action_space.sample()
        state, reward, _, _ = env.step(action)
        rewards.append(reward)
        grid = state[0]
        image = state[1]

    if reward == 1:
        print('reward', reward, e_i)
        e_i += 1
