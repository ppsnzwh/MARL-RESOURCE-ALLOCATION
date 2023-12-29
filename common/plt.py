import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator


def plt_cpu_ratio(result, x_labels = ["qmix", "vdn", "coma", "random", "greed"]):
    cloud1_list = []
    cloud2_list = []
    cloud3_list = []
    for i in range(len(x_labels)):
        cloud1_list.append(result[i]["cpu_ratio"][0])
        cloud2_list.append(result[i]["cpu_ratio"][1])
        cloud3_list.append(result[i]["cpu_ratio"][2])
    # x轴坐标, size=5, 返回[0, 1, 2, 3, 4]
    x = np.arange(len(x_labels))
    # 用第1组...替换横坐标x的值
    # 有a/b/c三种类型的数据，n设置为3
    total_width, n = 0.8, 3
    # 每种类型的柱状图宽度
    width = total_width / n
    # 重新设置x轴的坐标
    x = x - (total_width - width) / n
    plt.ylim(0, 1)  # 仅设置y轴坐标范围
    # 画柱状图
    plt.bar(x - width, cloud1_list, width=width, label="cloud_1")
    plt.bar(x, cloud2_list, width=width, label="cloud_2")
    plt.bar(x + width, cloud3_list, width=width, label="cloud_3")
    # x_labels = ["qmix", "vdn", "coma", "random", "greed"]
    plt.xticks(x, x_labels)
    y_label = 'cpu ratio /%'
    plt.ylabel(y_label)
    plt.xlabel('Algorithm')

    # 显示图例
    plt.legend()
    # 显示柱状图
    plt.show()
    plt.close()


def plt_memory_ratio(result    , x_labels = ["qmix", "vdn", "coma", "random", "greed"]):
    cloud1_list = []
    cloud2_list = []
    cloud3_list = []
    for i in range(len(x_labels)):
        cloud1_list.append(result[i]["memory_ratio"][0])
        cloud2_list.append(result[i]["memory_ratio"][1])
        cloud3_list.append(result[i]["memory_ratio"][2])
    # x轴坐标, size=5, 返回[0, 1, 2, 3, 4]
    x = np.arange(len(x_labels))
    # 用第1组...替换横坐标x的值
    # 有a/b/c三种类型的数据，n设置为3
    total_width, n = 0.9, 3
    # 每种类型的柱状图宽度
    width = total_width / n
    # 重新设置x轴的坐标
    x = x - (total_width - width) / n
    plt.ylim(0, 1)  # 仅设置y轴坐标范围
    # 画柱状图
    plt.bar(x - width, cloud1_list, width=width, label="cloud_1")
    plt.bar(x, cloud2_list, width=width, label="cloud_2")
    plt.bar(x + width, cloud3_list, width=width, label="cloud_3")
    # x_labels = ["qmix", "vdn", "coma", "random", "greed"]
    plt.xticks(x, x_labels)
    y_label = 'memory ratio /%'
    plt.ylabel(y_label)
    plt.xlabel('Algorithm')
    # 显示图例
    plt.legend()
    # 显示柱状图
    plt.show()
    plt.close()


def plt_price(result,     x_labels = ["qmix", "vdn", "coma", "random", "greed"]):
    price_list = []
    for i in range(len(x_labels)):
        temp_reward = result[i]["price"]
        price_list.append(temp_reward)
    # x轴坐标, size=5, 返回[0, 1, 2, 3, 4]
    x = np.arange(len(x_labels))
    color = ['darkgreen', 'cyan', 'salmon', 'black', 'lightgreen']
    # 画柱状图
    plt.bar(x, price_list, color=color)
    # x_labels = ["qmix", "vdn", "coma", "random", "greed"]
    plt.xticks(x, x_labels)
    y_label = 'price values /$'
    plt.ylabel(y_label)
    plt.xlabel('Algorithm')
    # 显示图例
    plt.legend()
    # 显示柱状图
    plt.show()
    plt.close()


def plt_social(result,     x_labels = ["qmix", "vdn", "coma", "random", "greed"]):
    welfare_list = []
    for i in range(len(x_labels)):
        temp_reward = result[i]["social_welfare"]
        welfare_list.append(temp_reward)
    # x轴坐标, size=5, 返回[0, 1, 2, 3, 4]
    x = np.arange(len(x_labels))
    color = ['darkgreen', 'cyan', 'salmon', 'black', 'lightgreen']
    # 画柱状图
    plt.bar(x, welfare_list, color=color)
    # x_labels = ["qmix", "vdn", "coma", "random", "greed"]
    plt.xticks(x, x_labels)
    y_label = 'social welfare values /$'
    plt.ylabel(y_label)
    plt.xlabel('Algorithm')
    # 显示图例
    plt.legend()
    # 显示柱状图
    plt.show()
    plt.close()


def plt_reward(result  ,  x_labels = ["qmix", "vdn", "coma", "random", "greed"]):
    if len(result) > 0:
        reward_list = []
        for i in range(len(x_labels)):
            temp_reward = result[i]["reward"]
            reward_list.append(temp_reward)
        # x轴坐标, size=5, 返回[0, 1, 2, 3, 4]
        x = np.arange(len(x_labels))
        color = ['darkgreen', 'cyan', 'salmon', 'black', 'lightgreen']
        # 画柱状图
        plt.bar(x, reward_list, color=color)
        # x_labels = ["qmix", "vdn", "coma", "random", "greed"]
        plt.xticks(x, x_labels)
        y_label = 'reward'
        plt.ylabel(y_label)
        plt.xlabel('Algorithm')
        # 显示图例
        plt.legend()
        # 显示柱状图
        plt.show()
        plt.close()



def plt_train(list):
    x = range(len(list))
    y = list
    # plt.subplot(2, 1, 2)
    plt.plot(x, y,  'bo--', alpha=0.3, linewidth=1, label='reward')
    plt.xlabel('train step * 500')
    y_label = 'train rewards'
    plt.ylabel(y_label)
    plt.show()
    plt.close()

def plt_evaluate(list):
    x = range(len(list))
    y = list
    # plt.subplot(2, 1, 2)
    plt.plot(x, y,  'bo--', alpha=1, linewidth=1, label='reward')
    plt.xlabel('evaluate step *20000')
    y_label = 'evaluate rewards'
    plt.ylabel(y_label)
    plt.show()
    plt.close()
