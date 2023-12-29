import argparse


def get_common_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    # parser.add_argument('--difficulty', type=str, default='7', help='the difficulty of the game')
    # parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
    parser.add_argument('--map', type=str, default='resource_allocation', help='the map of the game')
    parser.add_argument('--generate', type=bool, default=False, help='whether generate data')
    parser.add_argument('--generate_num', type=int, default=500, help='How many pieces of data are generated')
    parser.add_argument('--batch_num', type=int, default=5, help='How many pieces of data running per second')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='absolute path to save the replay')
    # The alternative algorithms are vdn, coma, qmix,
    parser.add_argument('--alg', type=str, default='qmix', help='the algorithm to train the agent')
    # parser.add_argument('--n_steps', type=int, default=2000000, help='total time steps')
    parser.add_argument('--n_steps', type=int, default=1000000, help='total time steps')
    parser.add_argument('--n_episodes', type=int, default=1, help='the number of episodes before once training')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--evaluate_cycle', type=int, default=200000, help='how often to evaluate the model')
    parser.add_argument('--train_cycle', type=int, default=300000, help='how often to train the model')
    parser.add_argument('--evaluate_epoch', type=int, default=32, help='number of the epoch to evaluate the agent')
    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--evaluate', type=bool, default=False, help='whether to evaluate the model')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')
    parser.add_argument('--play_game', type=bool, default=False, help='whether play the game')
    parser.add_argument('--game_over', type=int, default=3, help='how many error steps to game over')
    args = parser.parse_args()
    return args


def get_cloud1_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=1)
    parser.add_argument('--cpu', type=int, default=80)
    parser.add_argument('--cpu_price', type=int, default=2.2)
    parser.add_argument('--memory', type=int, default=120)
    parser.add_argument('--memory_price', type=int, default=0.8)

    args = parser.parse_args()
    return args


def get_cloud2_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=2)
    parser.add_argument('--cpu', type=int, default=100)
    parser.add_argument('--cpu_price', type=int, default=2)
    parser.add_argument('--memory', type=int, default=100)
    parser.add_argument('--memory_price', type=int, default=1)
    args = parser.parse_args()
    return args


def get_cloud3_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=3)
    parser.add_argument('--cpu', type=int, default=120)
    parser.add_argument('--cpu_price', type=int, default=1.8)
    parser.add_argument('--memory', type=int, default=80)
    parser.add_argument('--memory_price', type=int, default=1.2)
    args = parser.parse_args()
    return args

def get_env_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_actions', type=int, default=2)
    parser.add_argument('--n_agents', type=int, default=3)
    parser.add_argument('--state_shape', type=int, default=8)
    parser.add_argument('--obs_shape', type=int, default=4)
    parser.add_argument('--episode_limit', type=int, default=500)
    args = parser.parse_args()
    return args

def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    args.lr = 5e-4

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    # anneal_steps = 50000
    anneal_steps = 1500000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps  #0.000019
    # args.epsilon_anneal_scale = 'episode'
    args.epsilon_anneal_scale = 'step'

    # the number of the train steps in one epoch
    args.train_steps = 1

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(5e3)

    # how often to save the model
    args.save_cycle = 500

    # how often to update the target_net
    args.target_update_cycle = 200

    # QTRAN lambda
    args.lambda_opt = 1
    args.lambda_nopt = 1

    # prevent gradient explosion
    args.grad_norm_clip = 10

    # MAVEN
    args.noise_dim = 16
    args.lambda_mi = 0.001
    args.lambda_ql = 1
    args.entropy_coefficient = 0.001
    return args

def get_coma_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args

if __name__ == "__main__":
    args =get_env_args()
    args = get_mixer_args(args)
    print(args)