import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time
from decimal import *
from generate.generate_data import generate_data
from generate.load_data import load_data
from requirement.requirement import Req
from common.spacial_utils import noramlization, obs_noramlization, state_noramlization


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit  # 70
        self.n_actions = args.n_actions  # 2
        self.n_agents = args.n_agents  # 3
        self.state_shape = args.state_shape  # 8
        self.obs_shape = args.obs_shape  # 4
        self.args = args

        self.epsilon = args.epsilon  # 1
        self.anneal_epsilon = args.anneal_epsilon  # 1.8999999999999998e-05
        self.min_epsilon = args.min_epsilon  # 0.05
        print('Init RolloutWorker')

    @torch.no_grad()
    def generate_episode(self, episode_num=None, evaluate=False, data=None):
        if data is None:
            data = []
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        batch_num = self.args.batch_num
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        terminated = False
        time_step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))  # shape:(3,2)
        self.agents.policy.init_hidden(1)

        # epsilon 探索率 训练时从1开始递减，验证时为0不探索
        epsilon = 0 if evaluate else self.epsilon

        if self.args.epsilon_anneal_scale == 'episode':
            # anneal_epsilon= 0.000019    min_epsilon=0.05
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        batch_list = []
        temp_list = []
        num = 0
        # 将数据五个一批放入
        for k in range(len(data)):
            if num == batch_num:
                num = 0
                batch_list.append(temp_list)
                temp_list = []
            if num < batch_num :
                temp_list.append(data[k])
                num += 1
            if k == len(data)-1:
                num = 0
                batch_list.append(temp_list)
                temp_list = []
        temp_num = 0
        temp_flag = 0
        while not terminated and time_step < self.episode_limit / batch_num:
            batch_data = batch_list[time_step]
            self.env.set_time_step(time_step)
            self.env.release_all_cloud_requirement()

            for j in range(batch_num):
                if terminated:
                    temp_num = 4-j + 1
                    temp_flag = 1
                    break
                req = batch_data[j]
                requirement = Req(req)

                # time.sleep(0.2)

                self.env.set_requirement(requirement)
                obs = self.env.get_obs()  # shape=(3,4)
                obs = obs_noramlization(obs)
                state = self.env.get_state()  # shape=(8,0)
                state = state_noramlization(state)

                actions, avail_actions, actions_onehot = [], [], []
                # 有多个智能体
                for agent_id in range(self.n_agents):
                    avail_action = [0, 1]
                    if agent_id == 0:
                        avail_action = self.env.cloud_1.avail_action(requirement, epsilon)
                    if agent_id == 1:
                        avail_action = self.env.cloud_2.avail_action(requirement, epsilon)
                    if agent_id == 2:
                        avail_action = self.env.cloud_3.avail_action(requirement, epsilon)
                    action = self.agents.choose_action( obs[agent_id], last_action[agent_id], agent_id, epsilon, avail_action, requirement=requirement)
                    # generate onehot vector of th action
                    action_onehot = np.zeros(self.args.n_actions)  # (2，)
                    action_onehot[action] = 1
                    actions.append(np.int(action))
                    actions_onehot.append(action_onehot)
                    avail_actions.append(avail_action)
                    last_action[agent_id] = action_onehot

                reward, terminated, info = self.env.step(actions)
                o.append(obs)
                s.append(state)
                u.append(np.reshape(actions, [self.n_agents, 1]))
                u_onehot.append(actions_onehot)
                avail_u.append(avail_actions)
                r.append([reward])
                terminate.append([terminated])
                padded.append([0.])
                episode_reward = float(Decimal(str(episode_reward)) + Decimal(str(reward)))
                if self.args.epsilon_anneal_scale == 'step':
                    epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
            time_step += 1

        if not evaluate:
            print("     the epsilon is：", epsilon)
        # last obs
        obs = self.env.get_obs()
        obs = obs_noramlization(obs)
        state = self.env.get_state()
        state = state_noramlization(state)
        o.append(obs)
        s.append(state)
        # 从左往右从数组第一个索引开始取
        o_next = o[1:]
        s_next = s[1:]
        # 从右往左从数组第一个索引开始取
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        # 获取最后一个工作的avail_action，因为target_q在训练中需要avail_action
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = [0, 1]
            if agent_id == 0:
                avail_action = self.env.cloud_1.avail_action(requirement, epsilon)
            if agent_id == 1:
                avail_action = self.env.cloud_2.avail_action(requirement, epsilon)
            if agent_id == 2:
                avail_action = self.env.cloud_3.avail_action(requirement, epsilon)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        if temp_flag == 0:
            # if step < self.episode_limit，padding
            for i in range(time_step * batch_num, self.episode_limit):
                o.append(np.zeros((self.n_agents, self.obs_shape)))
                u.append(np.zeros([self.n_agents, 1]))
                s.append(np.zeros(self.state_shape))
                r.append([0.])
                o_next.append(np.zeros((self.n_agents, self.obs_shape)))
                s_next.append(np.zeros(self.state_shape))
                u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
                avail_u.append(np.zeros((self.n_agents, self.n_actions)))
                avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
                padded.append([1.])
                terminate.append([1.])
        else:
            # if step < self.episode_limit，padding
            for i in range(time_step * batch_num - temp_num, self.episode_limit):
                o.append(np.zeros((self.n_agents, self.obs_shape)))
                u.append(np.zeros([self.n_agents, 1]))
                s.append(np.zeros(self.state_shape))
                r.append([0.])
                o_next.append(np.zeros((self.n_agents, self.obs_shape)))
                s_next.append(np.zeros(self.state_shape))
                u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
                avail_u.append(np.zeros((self.n_agents, self.n_actions)))
                avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
                padded.append([1.])
                terminate.append([1.])


        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()

        return episode, episode_reward, time_step


