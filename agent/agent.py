import numpy as np
import torch
from torch.distributions import Categorical


# Agent no communication
class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        if args.alg == 'vdn':
            from policy.vdn import VDN
            self.policy = VDN(args)
        elif args.alg == 'qmix':
            from policy.qmix import QMIX
            self.policy = QMIX(args)
        elif args.alg == 'coma':
            from policy.coma import COMA
            self.policy = COMA(args)
        elif args.alg == 'qmix_no_share_rnn':
            from policy.qmix_no_share_rnn import QMIX_NO_SHARE_RNN
            self.policy = QMIX_NO_SHARE_RNN(args)

        self.args = args

    def choose_action(self, obs, last_action, agent_num, epsilon, avail_action, maven_z=None, requirement=None):
        avail_actions_ind = np.nonzero(avail_action)[0]  # index of actions which can be choose
        if self.args.alg == 'qmix':
            if self.args.play_game:
                inputs = obs.copy()
                # transform agent_num to onehot vector
                agent_id = np.zeros(self.n_agents)
                agent_id[agent_num] = 1.
                if self.args.last_action:  # 选择动作时是否考虑最后一个动作
                    # np.hstack():在水平方向上平铺
                    inputs = np.hstack((inputs, last_action))  # (6,)
                if self.args.reuse_network:  # 是否为所有代理使用一个网络
                    inputs = np.hstack((inputs, agent_id))  # (9,)
                hidden_state = self.policy.eval_hidden[:, agent_num,
                               :]  # elf.policy.eval_hidden=(1,3,64)  hidden_state = torch.Size([1, 64])
                # transform the shape of inputs from (9,) to (1,9)  unsqueeze 扩展维度
                inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)  # (1,9)
                if self.args.cuda:
                    inputs = inputs.cuda()
                    hidden_state = hidden_state.cuda()
                q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs,
                                                                                         hidden_state)  # 第一个输出是q值，第二个输出是rnn的输出
                # choose action from q value
                if np.random.uniform() < epsilon:
                    action = np.random.choice(avail_actions_ind)  # action是一个整数
                else:
                    action = torch.argmax(q_value)
            else:
                inputs = obs.copy()  # (4,)
                # transform agent_num to onehot vector
                agent_id = np.zeros(self.n_agents)  # (3,)
                agent_id[agent_num] = 1.
                if self.args.last_action:  # 选择动作时是否考虑最后一个动作
                    # np.hstack():在水平方向上平铺
                    inputs = np.hstack((inputs, last_action))  # (6,)
                if self.args.reuse_network:  # 是否为所有代理使用一个网络
                    inputs = np.hstack((inputs, agent_id))  # (9,)
                hidden_state = self.policy.eval_hidden[:, agent_num,
                               :]  # elf.policy.eval_hidden=(1,3,64)  hidden_state = torch.Size([1, 64])
                # transform the shape of inputs from (9,) to (1,9)  unsqueeze 扩展维度
                inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)  # (1,9)
                avail_actions = torch.tensor(avail_action, dtype=torch.float32).unsqueeze(0)  # (1,2)
                if self.args.cuda:
                    inputs = inputs.cuda()
                    hidden_state = hidden_state.cuda()
                # get q value
                # q_value=torch.Size([1, 12])   eval_hidden=torch.Size([1, 5, 64])
                q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs,
                                                                                         hidden_state)  # 第一个输出是q值，第二个输出是rnn的输出                                                                   hidden_state)  # 第一个输出是q值，第二个输出是rnn的输出
                # choose action from q value
                q_value[avail_actions == 0.0] = - float("inf")
                if np.random.uniform() < epsilon:
                    action = np.random.choice(avail_actions_ind)  # action是一个整数
                else:
                    action = torch.argmax(q_value)
        elif self.args.alg == 'qmix_no_share_rnn':
            inputs = obs.copy()  # (4,)
            # transform agent_num to onehot vector
            agent_id = np.zeros(self.n_agents)  # (3,)
            agent_id[agent_num] = 1.
            if self.args.last_action:  # 选择动作时是否考虑最后一个动作
                # np.hstack():在水平方向上平铺
                inputs = np.hstack((inputs, last_action))  # (6,)
            if self.args.reuse_network:  # 是否为所有代理使用一个网络
                inputs = np.hstack((inputs, agent_id))  # (9,)
            hidden_state = self.policy.eval_hidden[:, agent_num,
                           :]  # elf.policy.eval_hidden=(1,3,64)  hidden_state = torch.Size([1, 64])
            # transform the shape of inputs from (9,) to (1,9)  unsqueeze 扩展维度
            inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)  # (1,9)
            avail_actions = torch.tensor(avail_action, dtype=torch.float32).unsqueeze(0)  # (1,2)
            if self.args.cuda:
                inputs = inputs.cuda()
                hidden_state = hidden_state.cuda()
            if agent_num == 0:
                # get q value
                # q_value=torch.Size([1, 12])   eval_hidden=torch.Size([1, 5, 64])
                q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs,
                                                                                         hidden_state)  # 第一个输出是q值，第二个输出是rnn的输出
            elif agent_num == 1:
                # get q value
                # q_value=torch.Size([1, 12])   eval_hidden=torch.Size([1, 5, 64])
                q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn1(inputs,
                                                                                          hidden_state)  # 第一个输出是q值，第二个输出是rnn的输出
            else:
                # get q value
                # q_value=torch.Size([1, 12])   eval_hidden=torch.Size([1, 5, 64])
                q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn2(inputs,
                                                                                          hidden_state)  # 第一个输出是q值，第二个输出是rnn的输出
            # choose action from q value
            q_value[avail_actions == 0.0] = - float("inf")
            if np.random.uniform() < epsilon:
                action = np.random.choice(avail_actions_ind)  # action是一个整数
            else:
                action = torch.argmax(q_value)
        else:
            inputs = obs.copy()  # (4,)
            # transform agent_num to onehot vector
            agent_id = np.zeros(self.n_agents)  # (3,)
            agent_id[agent_num] = 1.

            if self.args.last_action:  # 选择动作时是否考虑最后一个动作
                # np.hstack():在水平方向上平铺
                inputs = np.hstack((inputs, last_action))  # (6,)
            if self.args.reuse_network:  # 是否为所有代理使用一个网络
                inputs = np.hstack((inputs, agent_id))  # (9,)

            hidden_state = self.policy.eval_hidden[:, agent_num,
                           :]  # elf.policy.eval_hidden=(1,3,64)  hidden_state = torch.Size([1, 64])

            # transform the shape of inputs from (9,) to (1,9)  unsqueeze 扩展维度
            inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)  # (1,9)
            avail_actions = torch.tensor(avail_action, dtype=torch.float32).unsqueeze(0)  # (1,2)
            if self.args.cuda:
                inputs = inputs.cuda()
                hidden_state = hidden_state.cuda()

            # get q value
            # q_value=torch.Size([1, 12])   eval_hidden=torch.Size([1, 5, 64])
            # 第一个输出是q值，第二个输出是rnn的输出
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)
            # choose action from q value
            if self.args.alg == 'coma' or self.args.alg == 'central_v' or self.args.alg == 'reinforce':
                action = self._choose_action_from_softmax(q_value.cpu(), avail_actions, epsilon)
            else:
                q_value[avail_actions == 0.0] = - float("inf")
                if np.random.uniform() < epsilon:
                    action = np.random.choice(avail_actions_ind)  # action是一个整数
                else:
                    action = torch.argmax(q_value)
        return action

    def _choose_action_from_softmax(self, inputs, avail_actions, epsilon):
        """
        :param inputs: # q_value of all actions
        """
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[
            -1])  # num of avail_actions
        # 先将Actor网络的输出通过softmax转换成概率分布
        prob = torch.nn.functional.softmax(inputs, dim=-1)
        # add noise of epsilon
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        """
        不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
        会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
        """

        action = Categorical(prob).sample().long()
        return action

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        if max_episode_len == 0:  # 防止所有的episode都没有结束，导致terminated中没有1
            max_episode_len = self.args.episode_limit
        return max_episode_len

    def train(self, batch, train_step, epsilon=None, epsilon_reward=None):  # coma needs epsilon for training
        if epsilon_reward is None:
            epsilon_reward = []
        print("start train")
        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)  # max_episode_len=21
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step, epsilon_reward)
            print(self.args.alg, " model save success")
