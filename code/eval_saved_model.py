import argparse
import torch

from lunar_agent import Agent, DenseNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("fpath", action="store", dest="fpath")
    parser.add_argument(
        "eval_eps", action="store", dest="eval_eps", type=int, default=100
    )
    parser.add_argument("video", action="store", dest="video", type=bool, default=True)

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

    args = parser.parse_args()

    policy_net = DenseNet(8, 4).float().to(device)
    target_net = DenseNet(8, 4).float().to(device)
    try:
        target_net.load_state_dict(torch.load(args.fpath))
    except Exception as e:
        print("Failed to load state dict\n")
        print(e.message, e.args)
        return

    agent = Agent(params_dict, policy_net=policy_net, target_net=target_net)
    agent.eval_policy(args.eval_eps)

    if args.video:
        agent.make_video()


if __name__ == "__main__":
    main()
