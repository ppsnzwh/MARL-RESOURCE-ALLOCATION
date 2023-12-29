import numpy as np
import os
from common.rollout import RolloutWorker
from agent.agent import Agents
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
from common.write_txt import *

from generate.generate_data import generate_data
from generate.load_data import load_data


class Runner:
    def __init__(self, env, args, data_size=1):
        self.env = env

        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)

        if not args.evaluate and args.alg.find('coma') == -1:  # these 3 algorithms are on-poliy
            self.buffer = ReplayBuffer(args)  # 经验回放缓冲池
        self.args = args
        self.win_rates = []
        self.train_episode_rewards = []
        self.episode_rewards = []
        self.episode_prices = []
        self.episode_social_welfare = []
        self.data = []
        # load data
        data = load_data(data_size)

        self.data = data

        # 用来保存plt和pkl
        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map  # './result/qmix/resource_allocation'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        train_plt_steps = 1
        time_steps, train_steps, evaluate_steps = 0, 0, 0
        # '--n_steps', type=int, default=2000000, help='total time steps'
        test_step = 0

        while time_steps < self.args.n_steps:
            test_step += 1
            print('Run {}, time_steps {}'.format(num, time_steps))
            # //表示除法结果向下取整
            # '--evaluate_cycle', type=int, default=5000, help='how often to evaluate the model'
            if time_steps // self.args.evaluate_cycle > evaluate_steps:
                # 用来评估
                print("现在是第{}步，正在生产评估结果".format(time_steps))

                episode_reward = self.evaluate()
                print(" 第", str(time_steps), "轮的评估,", self.args.alg, "算法的行动历史为：", self.env.action_history_list)
                print(" 第", str(time_steps), "轮评估，32次平均奖励为：", episode_reward)
                # print('win_rate is ', win_rate)
                self.episode_rewards.append(episode_reward)
                self.episode_prices.append(self.env.total_earnings)
                self.episode_social_welfare.append(self.env.total_social_welfare)
                self.plt(num)
                # self.plt_cpu_ratio(1)
                # self.plt_cpu_ratio(2)
                # self.plt_cpu_ratio(3)
                # self.plt_memory_ratio(1)
                # self.plt_memory_ratio(2)
                # self.plt_memory_ratio(3)
                evaluate_steps += 1
            episodes = []
            # 收集self.args.n_episodes个episodes
            # parser.add_argument('--n_episodes', type=int, default=1,
            # help='the number of episodes before once training')
            for episode_idx in range(self.args.n_episodes):  # n_episodes=1
                episode, r_, steps = self.rolloutWorker.generate_episode(episode_idx, False, self.data)

                print(" 第", str(time_steps), "轮训练,", self.args.alg, "算法的行动历史为：", self.env.action_history_list)
                counter = 0
                for i in self.env.action_history_list:
                    if i == -1:
                        # incrementing counter
                        counter = counter + 1
                print(" 第", str(time_steps), "轮的拒绝率为：",round(counter/len(self.env.action_history_list), 2))
                print(" 第", str(time_steps), "轮的训练奖励为：", r_)

                self.train_episode_rewards.append(r_)
                episodes.append(episode)
                time_steps += (steps * self.args.batch_num)
                # print(_)
            if time_steps % self.args.train_cycle == train_plt_steps:
                train_plt_steps += 1
                self.plt_train()
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find('reinforce') > -1:
                self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                train_steps += 1
            else:
                self.buffer.store_episode(episode_batch)
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    self.agents.train(mini_batch, train_steps,  epsilon_reward=self.episode_rewards)
                    train_steps += 1

        # 全部训练完了在执行最后一次评估
        episode_reward = self.evaluate()
        self.episode_rewards.append(episode_reward)
        self.show_plt(num)
        self.plt_resource_ratio()
        self.plt_price(num)
        write_list_data(self.args.alg, self.train_episode_rewards, self.args.time)
        write_list_evaluate_data(self.args.alg, self.episode_rewards, self.args.time)
        write_list_price_data(self.args.alg, self.episode_prices, self.args.time)
        write_list_social_data(self.args.alg, self.episode_social_welfare, self.args.time)

    def evaluate(self):
        episode_rewards = 0

        # evaluate_epoch 表示 number of the epoch to evaluate the agent 初始值32
        for epoch in range(self.args.evaluate_epoch):  # evaluate_epoch=32
            _, episode_reward, _ = self.rolloutWorker.generate_episode(epoch, evaluate=True, data=self.data)
            episode_rewards += episode_reward
        return episode_rewards / self.args.evaluate_epoch

    def evaluate_result(self):
        episode_rewards = 0
        cloud1_cpu_ratio_avg = 0
        cloud2_cpu_ratio_avg = 0
        cloud3_cpu_ratio_avg = 0
        cloud1_memory_use_ratio_history = 0
        cloud2_memory_use_ratio_history = 0
        cloud3_memory_use_ratio_history = 0

        temp_price = 0
        temp_social_welfare = 0

        # evaluate_epoch 表示 number of the epoch to evaluate the agent 初始值32
        for epoch in range(self.args.evaluate_epoch):  # evaluate_epoch=32
            _, episode_reward, _ = self.rolloutWorker.generate_episode(epoch, evaluate=True, data=self.data)
            episode_rewards += episode_reward
            temp_price += self.env.total_earnings
            temp_social_welfare += self.env.total_social_welfare

            cloud1_cpu_ratio_avg += sum(self.env.cloud_1.cpu_use_ratio_history) / len(
                self.env.cloud_1.cpu_use_ratio_history)
            cloud2_cpu_ratio_avg += sum(self.env.cloud_2.cpu_use_ratio_history) / len(
                self.env.cloud_2.cpu_use_ratio_history)
            cloud3_cpu_ratio_avg += sum(self.env.cloud_3.cpu_use_ratio_history) / len(
                self.env.cloud_3.cpu_use_ratio_history)
            cloud1_memory_use_ratio_history += sum(self.env.cloud_1.memory_use_ratio_history) / len(
                self.env.cloud_1.memory_use_ratio_history)
            cloud2_memory_use_ratio_history += sum(self.env.cloud_2.memory_use_ratio_history) / len(
                self.env.cloud_2.memory_use_ratio_history)
            cloud3_memory_use_ratio_history += sum(self.env.cloud_3.memory_use_ratio_history) / len(
                self.env.cloud_3.memory_use_ratio_history)

        reward = episode_rewards / self.args.evaluate_epoch
        cpu_ratio = [cloud1_cpu_ratio_avg/self.args.evaluate_epoch, cloud2_cpu_ratio_avg/self.args.evaluate_epoch, cloud3_cpu_ratio_avg/self.args.evaluate_epoch]
        memory_ratio = [cloud1_memory_use_ratio_history/self.args.evaluate_epoch, cloud2_memory_use_ratio_history/self.args.evaluate_epoch, cloud3_memory_use_ratio_history/self.args.evaluate_epoch]
        price = temp_price / self.args.evaluate_epoch
        social_welfare = temp_social_welfare / self.args.evaluate_epoch
        result = {'reward': reward,
                  "cpu_ratio": cpu_ratio,
                  "memory_ratio": memory_ratio,
                  "price": price,
                  "social_welfare": social_welfare
                }
        return result

    def plt(self, num):

        plt.ylim([0, 105])
        plt.cla()
        x = range(len(self.episode_rewards))
        y = self.episode_rewards
        # plt.subplot(2, 1, 2)
        plt.plot(x, y,  'bo--', alpha=0.5, linewidth=1, label='reward')
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')
        # plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
        # np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)
        plt.show()
        plt.close()

    def plt_train(self):

        plt.cla()
        x = range(len(self.train_episode_rewards))
        y = self.train_episode_rewards
        # plt.subplot(2, 1, 2)
        plt.plot(x, y,  'bo--', alpha=0.5, linewidth=1, label='reward')
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        y_label = self.args.alg + 'train_episode_rewards'
        plt.ylabel(y_label)
        plt.show()
        plt.close()

    def plt_price(self, num):
        x1 = range(len(self.episode_prices))
        x2 = range(len(self.episode_social_welfare))
        y1 = self.episode_prices
        y2 = self.episode_social_welfare
        plt.plot(x1, y1,  'bo--', alpha=0.5, linewidth=1, label='prices')  # 'bo-'表示蓝色实线，数据点实心原点标注
        plt.plot(x2, y2, 'r*--', alpha=0.5, linewidth=1, label='social_welfare')  # 'bo-'表示蓝色实线，数据点实心原点标注
        ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，

        plt.xlabel('step')
        y_label = self.args.alg + 'price_num'
        plt.ylabel(y_label)

        # plt.savefig(self.save_path + '/plt_{}_price.png'.format(num), format='png')
        # np.save(self.save_path + '/episode_price_{}'.format(num), self.episode_prices)
        # np.save(self.save_path + '/episode_social_welfare_{}'.format(num), self.episode_social_welfare)
        plt.legend()  # 显示上面的label
        plt.show()
        plt.close()

    def plt_resource_ratio(self):
        cpu_list = []
        cloud1_cpu_ratio_avg = sum(self.env.cloud_1.cpu_use_ratio_history) / len(self.env.cloud_1.cpu_use_ratio_history)
        cloud2_cpu_ratio_avg = sum(self.env.cloud_2.cpu_use_ratio_history) / len(self.env.cloud_2.cpu_use_ratio_history)
        cloud3_cpu_ratio_avg = sum(self.env.cloud_3.cpu_use_ratio_history) / len(self.env.cloud_3.cpu_use_ratio_history)
        cpu_list.append(cloud1_cpu_ratio_avg)
        cpu_list.append(cloud2_cpu_ratio_avg)
        cpu_list.append(cloud3_cpu_ratio_avg)

        memory_list = []
        cloud1_memory_use_ratio_history = sum(self.env.cloud_1.memory_use_ratio_history) / len(
            self.env.cloud_1.memory_use_ratio_history)
        cloud2_memory_use_ratio_history = sum(self.env.cloud_2.memory_use_ratio_history) / len(
            self.env.cloud_2.memory_use_ratio_history)
        cloud3_memory_use_ratio_history = sum(self.env.cloud_3.memory_use_ratio_history) / len(
            self.env.cloud_3.memory_use_ratio_history)
        memory_list.append(cloud1_memory_use_ratio_history)
        memory_list.append(cloud2_memory_use_ratio_history)
        memory_list.append(cloud3_memory_use_ratio_history)

        # x轴坐标, size=5, 返回[0, 1, 2, 3, 4]
        x = np.arange(3)

        # 用第1组...替换横坐标x的值
        # 有a/b/c三种类型的数据，n设置为3
        total_width, n = 0.8, 3
        # 每种类型的柱状图宽度
        width = total_width / n

        # 重新设置x轴的坐标
        x = x - (total_width - width) / 2

        plt.ylim(0, 1)  # 仅设置y轴坐标范围
        # 画柱状图
        plt.bar(x, cpu_list, width=width, label="cpu_ratio")
        plt.bar(x + width, memory_list, width=width, label="memory_ratio")
        x_labels = ["cloud_1", "cloud_2", "cloud_3"]
        plt.xticks(x, x_labels)
        y_label = self.args.alg + 'resource_ratio'
        plt.ylabel(y_label)

        # 显示图例
        plt.legend()
        # 显示柱状图
        plt.show()
        plt.close()

    def plt_cpu_ratio(self, id):
        # plt.subplot(2, 1, 2)
        if id == 1:
            plt.plot(range(len(self.env.cloud_1.cpu_use_ratio_history)), self.env.cloud_1.cpu_use_ratio_history, 'b--', alpha=0.5, linewidth=1, label='c1_cpu')
        elif id == 2:
            plt.plot(range(len(self.env.cloud_2.cpu_use_ratio_history)), self.env.cloud_2.cpu_use_ratio_history, 'r--', alpha=0.5, linewidth=1, label='c2_cpu')
        else:
            plt.plot(range(len(self.env.cloud_3.cpu_use_ratio_history)), self.env.cloud_3.cpu_use_ratio_history, 'p--', alpha=0.5, linewidth=1, label='c3_cpu')
        plt.xlabel('step')
        y_label = self.args.alg + 'cpu_ratio'
        plt.ylabel(y_label)

        plt.show()
        plt.close()

    def plt_memory_ratio(self, id):
        # plt.subplot(2, 1, 2)
        if id == 1:
            plt.plot(range(len(self.env.cloud_1.memory_use_ratio_history)), self.env.cloud_1.memory_use_ratio_history, 'b--', alpha=0.5, linewidth=1, label='c1_memory')
        elif id == 2:
            plt.plot(range(len(self.env.cloud_2.memory_use_ratio_history)), self.env.cloud_2.memory_use_ratio_history, 'r--', alpha=0.5, linewidth=1, label='c2_memory')
        else:
            plt.plot(range(len(self.env.cloud_3.memory_use_ratio_history)), self.env.cloud_3.memory_use_ratio_history, 'p--', alpha=0.5, linewidth=1, label='c23memory')
        plt.xlabel('step')
        y_label = self.args.alg + 'memory_ratio'
        plt.ylabel(y_label)

        plt.show()
        plt.close()

    def show_plt(self, num):

        # plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards,  'bo--', alpha=0.5, linewidth=1, label='prices')
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        y_label = self.args.alg + 'episode_rewards'
        plt.ylabel(y_label)
        plt.show()
        # plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
        # np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)
        plt.close()








