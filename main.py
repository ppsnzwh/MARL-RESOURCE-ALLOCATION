import datetime

import numpy as np
from matplotlib import pyplot as plt

from common.argument import get_common_args, get_mixer_args, get_coma_args
from common.plt import *
from common.write_txt import write_dict_data
from generate.generate_data import generate_data
from policy.random import random
from policy.greed import greed
from runner import Runner
from env.environment import Env
from smac.env import StarCraft2Env

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



if __name__ == '__main__':
    args = get_common_args()
    # args = get_mixer_args(args)
    env = Env(args)
    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]  # 2
    args.n_agents = env_info["n_agents"]  # 3
    args.state_shape = env_info["state_shape"]  # 8
    args.obs_shape = env_info["obs_shape"]  # 3
    args.episode_limit = env_info["episode_limit"]  # 3

    i = datetime.datetime.now()  # 获取当前时间
    time = str(i.month) + str(i.day) + str(i.hour) + str(i.minute) + str(i.second)
    args.time = time
    result = []
    env.reset()
    temp_result = {}
    if args.generate:
        generate_data(args.generate_num)

    # 测试数据大小结果
    # args.alg = 'qmix'
    # args = get_mixer_args(args)
    # # args.alg里的算法
    # for i in range(3):
    #     temp_result = []
    #     runner = Runner(env, args, data_size=i)
    #     if not args.evaluate:
    #         runner.run(i)
    #         temp_result = runner.evaluate_result()
    #         write_dict_data(args.alg, temp_result, args.time)
    #     else:
    #         temp_result = runner.evaluate_result()
    #         write_dict_data(args.alg, temp_result, args.time)
    #     result.append(temp_result)
    #     env.close()
    # print(result)

    # rnn model not share argument
    # args.alg = 'qmix_no_share_rnn'
    # args = get_mixer_args(args)
    # # args.alg里的算法
    # runner = Runner(env, args)
    # if not args.evaluate:
    #     runner.run(i)
    #     temp_result = runner.evaluate_result()
    #     write_dict_data(args.alg, temp_result, args.time)
    # else:
    #     temp_result = runner.evaluate_result()
    #     write_dict_data(args.alg, temp_result, args.time)
    # result.append(temp_result)
    # env.close()
    # print(result)

    for i in range(5):
        env.reset()
        temp_result = {}
        if i == 0:
            args.alg = 'qmix'
            args = get_mixer_args(args)
            # args.alg里的算法
            runner = Runner(env, args)
            if not args.evaluate:
                runner.run(i)
                temp_result = runner.evaluate_result()
                write_dict_data(args.alg, temp_result, args.time)
            else:
                temp_result = runner.evaluate_result()
                write_dict_data(args.alg, temp_result, args.time)
            result.append(temp_result)
            env.close()
        elif i == 1:
            args.alg = 'vdn'
            args = get_mixer_args(args)
            # args.alg里的算法
            runner = Runner(env, args)
            if not args.evaluate:
                runner.run(i)
                temp_result = runner.evaluate_result()
                write_dict_data(args.alg, temp_result, args.time)
            else:
                temp_result = runner.evaluate_result()
                write_dict_data(args.alg, temp_result, args.time)
            result.append(temp_result)
            env.close()
        elif i == 2:
            args.alg = 'coma'
            args = get_coma_args(args)
            # args.alg里的算法
            runner = Runner(env, args)
            if not args.evaluate:
                runner.run(i)
                temp_result = runner.evaluate_result()
                write_dict_data(args.alg, temp_result, args.time)
            else:
                temp_result = runner.evaluate_result()
                write_dict_data(args.alg, temp_result, args.time)
            result.append(temp_result)
            env.close()
        elif i == 3:
            args.alg = 'random'
            # 随机算法
            random = random(env, args)
            temp_result = random.generate_episode()
            write_dict_data(args.alg, temp_result, args.time)
            result.append(temp_result)
            env.close()
        elif i == 4:
            args.alg = 'greed'
            # 在线贪心算法
            greed = greed(env, args)
            temp_result = greed.generate_episode()
            write_dict_data(args.alg, temp_result, args.time)
            result.append(temp_result)
            env.close()
        else:
            args.alg = 'qmix_no_share_rnn'
            args = get_mixer_args(args)
            # args.alg里的算法
            runner = Runner(env, args)
            if not args.evaluate:
                runner.run(i)
                temp_result = runner.evaluate_result()
                write_dict_data(args.alg, temp_result, args.time)
            else:
                write_dict_data(args.alg, temp_result, args.time)
            result.append(temp_result)
            env.close()
    plt_reward(result)
    plt_price(result)
    plt_social(result)
    plt_cpu_ratio(result)
    plt_memory_ratio(result)
