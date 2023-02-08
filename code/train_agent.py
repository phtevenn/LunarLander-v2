import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt

from typing_extensions import OrderedDict
from lunar_agent import Agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DenseNet(nn.Module):
    def __init__(self, state_len, n_actions):
        super(DenseNet, self).__init__()
        self.network = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(state_len, 64)),
                    ("relu1", nn.ReLU()),
                    ("fc2", nn.Linear(64, 64)),
                    ("relu2", nn.ReLU()),
                    ("output", nn.Linear(64, n_actions)),
                ]
            )
        )

    def forward(self, x):
        return self.network(x)


def get_agent(params_dict):
    policy_net = DenseNet(8, 4).float().to(device)
    target_net = DenseNet(8, 4).float().to(device)
    target_net.load_state_dict(policy_net.state_dict())

    agent = Agent(params_dict, policy_net, target_net)
    return agent


def train_and_plot(
    agent, episodes=1000, eval_eps=100, model_name="", soft_update=False
):
    scores = agent.train(episodes, soft_update=soft_update)
    eval_scores = agent.eval_policy(eval_eps)

    plt.plot(scores)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(model_name + " training")
    plt.show()

    plt.boxplot(eval_scores)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(model_name + " optimal policy performance")
    plt.show()

    return eval_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    params_dict = {
        "LR": 3e-4,
        "WD": 0,
        "BUFFER_SIZE": 10000,
        "BATCH_SIZE": 64,
        "TARGET_UPDATE_DELAY": 10,
        "MAX_STEPS": 1000,
        "EPSILON_INIT": 0.9,
        "EPSILON_END": 0.03,
        "EPSILON_DECAY": 0.005,
        "GAMMA": 0.99,
        "TAU": 0.001,
    }

    parser.add_argument("-lr", action="store", dest="lr", type=float)
    parser.add_argument(
        "-buffer_size", action="store", dest="buffer_size", type=int
    )
    parser
    parser.add_argument("-max_steps", action="store", dest="max_steps", type=int)
    parser.add_argument("-eps_init", action="store", dest="eps_init", type=float)
    parser.add_argument("-eps_end", action="store", dest="eps_end", type=float)
    parser.add_argument("-eps_decay", action="store", dest="eps_decay", type=float)
    parser.add_argument("-gamma", action="store", dest="gamma", type=float)
    parser.add_argument(
        "-target_update", action="store", dest="target_update", type=int
    )
    parser.add_argument(
        "-episodes", action="store", dest="episodes", type=int, default=1000
    )
    parser.add_argument(
        "-eval_episodes", action="store", dest="eval_episodes", type=int, default=100
    )
    parser.add_argument("-name", action="store", dest="fname")
    parser.add_argument(
        "-soft_update", action="store", dest="soft_update", type=bool, default=True
    )
    parser.add_argument(
        "-video", action="store", dest="video", type=bool, default=True
    )

    args = parser.parse_args()
    params_dict["LR"] = args.lr
    params_dict["BUFFER_SIZE"] = args.buffer_size
    params_dict["MAX_STEPS"] = args.max_steps
    params_dict["EPSILON_INIT"] = args.eps_init
    params_dict["EPSILON_END"] = args.eps_end
    params_dict["EPSILON_DECAY"] = args.eps_decay
    params_dict["GAMMA"] = args.gamma
    params_dict["TARGET_UPDATE_DELAY"] = args.target_update

    agent = get_agent(params_dict)
    train_and_plot(
        agent,
        episodes=args.episodes,
        eval_eps=args.eval_episodes,
        model_name=args.fname,
        soft_update=args.soft_update,
    )
    agent.save_model_checkpoint(fname=args.fname)
    agent.make_video(args.fname)
