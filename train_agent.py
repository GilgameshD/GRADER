'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2021-12-21 11:57:44
LastEditTime: 2022-12-28 13:00:17
Description: 
'''

from env import ColorChangingRL
from agents import SAC
from grader import GRADER
import numpy as np
from utils import load_config
import copy
import argparse
import time
import os
np.set_printoptions(linewidth=np.inf)


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--mode', type=str, required=True, choices=['IID', 'OOD-S'], help='IID means i.i.d. samples and OOD-S means spurious correlation')
parser.add_argument('--agent', type=str, default='GRADER', choices=['GRADER', 'SAC'])
parser.add_argument('--grader_model', type=str, default='mlp', choices=['causal', 'full', 'mlp', 'gnn'], help='type of model used in GRADER')

parser.add_argument('--env', type=str, default='chemistry', help='name of environment')
parser.add_argument('--graph', type=str, default='chain', choices=['collider', 'chain', 'full', 'jungle'], help='type of groundtruth graph in chemistry')
args = parser.parse_args()

# environment parameters
if args.env == 'chemistry':
    num_steps = 10
    movement = 'Static' # Dynamic, Static
    if args.mode == 'IID':
        num_objects = 5
        num_colors = 5
    else:
        num_objects = 5
        num_colors = 5
    width = 5
    height = 5
    graph = args.graph + str(num_objects) # chain, full
    env = ColorChangingRL(
        test_mode=args.mode, 
        render_type='shapes', 
        num_objects=num_objects, 
        num_colors=num_colors, 
        movement=movement, 
        max_steps=num_steps
    )
    env.set_graph(graph)

    config = load_config(config_path="config/chemistry_config.yaml")
    agent_config = config[args.agent]
    env_params = {
        'action_dim': env.action_space.n,
        'num_colors': env.num_colors,
        'num_objects': env.num_objects,
        'width': env.width,
        'height': env.height,
        'state_dim': env.num_colors * env.num_objects * env.width * env.height * 2,
        'goal_dim': env.num_colors * env.num_objects * env.width * env.height,
        'adjacency_matrix': env.adjacency_matrix, # store the graph 
    }
    episode = 200
    test_episode = 100
else:
    raise ValueError('Wrong environment name')
env_params['env_name'] = args.env
agent_config['env_params'] = env_params
save_path = os.path.join('./log', args.exp_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

render = False
trails = 10
test_interval = 10
save_interval = 10000


if __name__ == '__main__':
    for t_i in range(trails):
        # create agent
        if args.agent == 'GRADER':
            agent_config['grader_model'] = args.grader_model
            agent = GRADER(agent_config)
        elif args.agent == 'SAC':
            agent = SAC(agent_config)

        save_gif_count = 0
        test_reward = []
        train_reward = []
        for e_i in range(episode):
            state = env.reset(stage='train')
            done = False
            one_train_reward = 0
            while not done:
                action = agent.select_action(env, state, False)
                next_state, reward, done, info = env.step(action)
                one_train_reward += reward

                if agent.name in ['SAC']: 
                    agent.store_transition([state, action, reward, next_state, done])
                    agent.train()
                elif agent.name in ['GRADER']:
                    agent.store_transition([state, action, next_state])

                state = copy.deepcopy(next_state)

            if agent.name in ['GRADER']: 
                agent.train()
            train_reward.append(one_train_reward)

            # save model
            if (e_i+1) % save_interval == 0:
                agent.model_id = e_i + 1
                agent.save_model()

            if (e_i+1) % test_interval == 0:
                test_reward_mean = []
                for t_j in range(test_episode):
                    state = env.reset(stage='test')
                    done = False
                    total_reward = 0
                    step_reward = []
                    while not done:
                        action = agent.select_action(env, state, True)
                        next_state, reward, done, info = env.step(action)

                        if render:
                            env.render()
                            time.sleep(0.05)

                        state = copy.deepcopy(next_state)
                        total_reward += reward
                        step_reward.append(reward)
                    test_reward_mean.append(total_reward)

                test_reward_mean = np.mean(test_reward_mean, axis=0)
                print('[{}/{}] [{}/{}] Test Reward: {}'.format(t_i, trails, e_i, episode, test_reward_mean))
                test_reward.append(test_reward_mean)
                np.save(os.path.join(save_path, 'tower.test.reward.'+str(t_i)+'.npy'), test_reward)
