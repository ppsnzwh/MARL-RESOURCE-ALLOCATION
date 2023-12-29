import numpy as np
from _decimal import Decimal
import matplotlib.pyplot as plt

from generate.generate_data import generate_data
from generate.load_data import load_data
from requirement.requirement import Req


class random:
    def __init__(self, env, arg):
        self.arg = arg
        self.env = env
        print("random init")

    def generate_episode(self):
        # generate_data(500)

        data = load_data(data_size=1)
        batch_num = 5

        self.env.reset()
        terminated = False
        time_step = 0
        episode_reward = 0  # cumulative rewards

        batch_list = []
        temp_list = []
        num = 0
        # 将数据五个一批放入
        for k in range(len(data)):
            if num == batch_num:
                num = 0
                batch_list.append(temp_list)
                temp_list = []
            if num < batch_num:
                temp_list.append(data[k])
                num += 1
            if k == len(data) - 1:
                num = 0
                batch_list.append(temp_list)
                temp_list = []

        while not terminated and time_step < 500 / batch_num:
            batch_data = batch_list[time_step]
            self.env.set_time_step(time_step)
            self.env.release_all_cloud_requirement()

            for j in range(batch_num):
                req = batch_data[j]
                requirement = Req(req)

                # time.sleep(0.2)

                self.env.set_requirement(requirement)
                flag = np.random.uniform() > 0.5
                actions = []
                if flag:
                    for agent_id in range(self.env.n_agents):
                        action = np.random.choice([0, 1], 1)
                        actions.append(action)

                    reward, terminated = self.env.random_step(actions, flag)
                    episode_reward += reward
                else:
                    reward, terminated = self.env.random_step(actions, flag)
                    episode_reward += reward



            time_step += 1
        # self.plt_resource_ratio()
        # self.plt_price(num)
        # print("random total_earnings:",self.env.total_earnings)
        # print("random total_social_welfare:",self.env.total_social_welfare)
        # print("random episode_reward:", episode_reward)

        cloud1_cpu_ratio_avg = sum(self.env.cloud_1.cpu_use_ratio_history) / len(self.env.cloud_1.cpu_use_ratio_history)
        cloud2_cpu_ratio_avg = sum(self.env.cloud_2.cpu_use_ratio_history) / len(self.env.cloud_2.cpu_use_ratio_history)
        cloud3_cpu_ratio_avg = sum(self.env.cloud_3.cpu_use_ratio_history) / len(self.env.cloud_3.cpu_use_ratio_history)

        cloud1_memory_use_ratio_history = sum(self.env.cloud_1.memory_use_ratio_history) / len(
            self.env.cloud_1.memory_use_ratio_history)
        cloud2_memory_use_ratio_history = sum(self.env.cloud_2.memory_use_ratio_history) / len(
            self.env.cloud_2.memory_use_ratio_history)
        cloud3_memory_use_ratio_history = sum(self.env.cloud_3.memory_use_ratio_history) / len(
            self.env.cloud_3.memory_use_ratio_history)

        result = {'reward': episode_reward,
                  "cpu_ratio": [cloud1_cpu_ratio_avg, cloud2_cpu_ratio_avg, cloud3_cpu_ratio_avg],
                  "memory_ratio": [cloud1_memory_use_ratio_history, cloud2_memory_use_ratio_history, cloud3_memory_use_ratio_history],
                  "price": self.env.total_earnings,
                  "social_welfare": self.env.total_social_welfare
                }
        return result

    def plt_price(self, num):
        x = np.arange(1)

        # 用第1组...替换横坐标x的值
        # 有a/b/c三种类型的数据，n设置为3
        total_width, n = 0.8, 3
        # 每种类型的柱状图宽度
        width = total_width / n

        # 重新设置x轴的坐标
        x = x - (total_width - width) / 2

        # 画柱状图
        plt.bar(x, self.env.total_earnings, width=width, label="total_earnings")
        plt.bar(x + width, self.env.total_social_welfare, width=width, label="total_social_welfare")
        x_labels = ["random"]
        plt.xticks(x, x_labels)
        plt.ylabel('random_resource_ratio')
        # 显示图例
        plt.legend()
        # 显示柱状图
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
        plt.ylabel('random_resource_ratio')
        # 显示图例
        plt.legend()
        # 显示柱状图
        plt.show()


        plt.close()

