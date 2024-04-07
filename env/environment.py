from common.argument import get_common_args, get_cloud1_args, get_cloud2_args, get_cloud3_args, get_env_args
from env.cloud_server import Cloud
from env import cloud_server
import numpy as np
from decimal import *

from requirement.requirement import Req


class Env:
    def __init__(self, args):
        self.requirement = None
        cloud1_args = get_cloud1_args()
        cloud2_args = get_cloud2_args()
        cloud3_args = get_cloud3_args()
        self.cloud_1 = Cloud(cloud1_args)
        self.cloud_2 = Cloud(cloud2_args)
        self.cloud_3 = Cloud(cloud3_args)

        env_args = get_env_args()
        self.n_actions = env_args.n_actions
        self.n_agents = env_args.n_agents
        self.state_shape = env_args.state_shape
        self.obs_shape = env_args.obs_shape
        self.episode_limit = env_args.episode_limit
        self.time_step = 0
        self.total_earnings = 0
        self.total_social_welfare = 0
        self.terminated_num = 0

        self.action_history_list = []
        self.price_history = []
        self.refuse_requirement_history = []

        self.args = args
        print("env initial success")

    def set_requirement(self, requirement):
        self.requirement = requirement

    def get_obs(self):
        obs = np.array([self.cloud_1.get_obs(self.requirement),
                self.cloud_2.get_obs(self.requirement),
                self.cloud_3.get_obs(self.requirement)])
        return obs

    def get_cloud_by_id(self,id):
        if id == 0:
            return self.cloud_1
        elif id == 1:
            return self.cloud_2
        elif id == 2:
            return self.cloud_3
        else:
            return False

    def get_all_cloud(self):
        return {"cloud_1": self.cloud_1,
                "cloud_2": self.cloud_2,
                "cloud_3": self.cloud_3}

    def get_env_info(self):
        return {"n_actions": self.n_actions,
                "n_agents": self.n_agents,
                "state_shape": self.state_shape,
                "obs_shape": self.obs_shape,
                "episode_limit": self.episode_limit}

    def get_state(self):
        array = [self.cloud_1.cpu, self.cloud_1.memory,
                 self.cloud_2.cpu, self.cloud_2.memory,
                 self.cloud_3.cpu, self.cloud_3.memory,
                 self.requirement.cpu, self.requirement.memory]
        state = np.array(array)
        state = state.astype(dtype=np.float32)
        return state

    def release_all_cloud_requirement(self):
        self.cloud_1.release_requirement()
        self.cloud_2.release_requirement()
        self.cloud_3.release_requirement()

    def compute_all_cloud_ratio(self):
        self.cloud_1.compute_resource_ratio()
        self.cloud_2.compute_resource_ratio()
        self.cloud_3.compute_resource_ratio()

    def step(self, actions):
        price_list = []
        id_list = []
        info = {}
        self.compute_all_cloud_ratio()
        for i, id in zip(actions,range(len(actions))):
            if i == 1:
                continue
            else:
                price = self.get_cloud_by_id(id).compute_price(self.requirement)
                id_list.append(id)
                price_list.append(price)
        info.update({"price_list": price_list})
        info.update({"requirement": price_list})
        info.update({"time_step": self.requirement})
        if len(price_list) <= 0:
            self.refuse_requirement_history.append(self.requirement)
            self.price_history.append(info)
            self.action_history_list.append(-1)
            return -0.1, False, info

        min_price = min(price_list)
        index = price_list.index(min_price)
        id = id_list[index]
        flag = self.get_cloud_by_id(id).execute_requirement(self.requirement)
        info = {"price_list": price_list, "final_price": min_price, "flag": flag,  "time_step": self.time_step}
        if not flag:
            self.price_history.append(info)
            self.action_history_list.append(-1)
            self.terminated_num += 1
            if self.terminated_num >= self.args.game_over:
                return -10, True, info
            else:
                return -1, False, info
        else:
            cloud_name = "null"
            if id == 0:
                cloud_name = "cloud1"
            elif id == 1:
                cloud_name = "cloud2"
            elif id == 2:
                cloud_name = "cloud3"
            info.update({"exec_cloud_name": cloud_name})
            if min_price <= 0:
                print("错误出价:", info)
                return -0.1, False, info
            self.total_earnings += min_price
            # self.total_social_welfare += self.requirement.bid - min_price
            self.total_social_welfare = float(Decimal(str(self.total_social_welfare)) + Decimal(str(self.requirement.bid)) - Decimal(str(min_price)))
            self.price_history.append(info)
            self.action_history_list.append(id)
            return 0.9, False, info

    def random_step(self, actions, flag):
        price_list = []
        id_list = []
        self.compute_all_cloud_ratio()

        if not flag:
            self.refuse_requirement_history.append(self.requirement)
            self.action_history_list.append(-1)
            return -0.1, False

        for i in range(len(actions)):
            if actions[i] == 1:
                continue
            else:
                cloud = self.get_cloud_by_id(i)
                flag = cloud.if_exec(self.requirement)
                if flag:
                    price = cloud.compute_price(self.requirement)
                    price_list.append(price)
                    id_list.append(i)

        if len(price_list) <= 0:
            self.refuse_requirement_history.append(self.requirement)
            self.action_history_list.append(-1)
            return -0.1, False

        min_price = min(price_list)
        index = price_list.index(min_price)
        id = id_list[index]
        flag = self.get_cloud_by_id(id).execute_requirement(self.requirement)
        if not flag:
            self.action_history_list.append(-1)
            return -0.1, False
        else:
            if min_price <= 0:
                print("错误出价")
            self.total_earnings += min_price
            # self.total_social_welfare += self.requirement.bid - min_price
            self.total_social_welfare = float(Decimal(str(self.total_social_welfare)) + Decimal(str(self.requirement.bid)) - Decimal(str(min_price)))
            return 0.9, False

    def set_time_step(self, time_step):
        self.time_step = time_step

        self.cloud_1.time_step = time_step
        self.cloud_2.time_step = time_step
        self.cloud_3.time_step = time_step

    def reset(self):
        cloud1_args = get_cloud1_args()
        cloud2_args = get_cloud2_args()
        cloud3_args = get_cloud3_args()
        self.cloud_1 = Cloud(cloud1_args)
        self.cloud_2 = Cloud(cloud2_args)
        self.cloud_3 = Cloud(cloud3_args)

        env_args = get_env_args()
        self.n_actions = env_args.n_actions
        self.n_agents = env_args.n_agents
        self.state_shape = env_args.state_shape
        self.obs_shape = env_args.obs_shape
        self.episode_limit = env_args.episode_limit
        self.total_earnings = 0
        self.total_social_welfare = 0

        self.price_history = []
        self.action_history_list = []
        self.refuse_requirement_history = []
        self.time_step = 0
        self.terminated_num = 0
        # print("env reset success")

    def close(self):
        print("env is closed")


if __name__ == '__main__':
    args = 0
    env = Env(args)

    requirement = Req(
        {"id": 1, "cpu": 10, "memory": 10, "demand_time": 1, "start_time": 0, "end_time": -1, "is_run": -1,
         "bid": 10, "reserve": -1})
    env.set_requirement(requirement)
    print(env.get_obs())




