import gym
import torch
import torch.nn.functional as F
import torch.optim as optim

import random
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from collections import namedtuple, deque

from gym.wrappers.monitoring import video_recorder

RANDOM_SEED = 42
SAVE_FPATH = "/output/"

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.style.use("ggplot")

Memory = namedtuple(
    "Entry", field_names=["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    def __init__(self, state_len, n_actions, buffer_size):
        self.state_len = state_len
        self.n_actions = n_actions
        self.buffer_size = buffer_size

        self.memory = deque(maxlen=buffer_size)

    def append(self, state, action, reward, next_state, done):
        self.memory.append(Memory(state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        states = (
            torch.from_numpy(np.vstack([e.state for e in batch if e is not None]))
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(np.vstack([e.action for e in batch if e is not None]))
            .long()
            .to(device)
        )
        rewards = (
            torch.from_numpy(np.vstack([e.reward for e in batch if e is not None]))
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(np.vstack([e.next_state for e in batch if e is not None]))
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in batch if e is not None]).astype(np.uint8)
            )
            .float()
            .to(device)
        )

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self, params, policy_net, target_net):
        self.env = gym.make("LunarLander-v2").unwrapped
        self.env.seed(RANDOM_SEED)
        self.n_actions = self.env.action_space.n
        self.state_len = self.env.observation_space.shape[0]

        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=params["LR"], weight_decay=params["WD"]
        )

        self.memory = ReplayBuffer(
            self.state_len, self.n_actions, params["BUFFER_SIZE"]
        )

        self.batch_size = params["BATCH_SIZE"]
        self.target_update_delay = params["TARGET_UPDATE_DELAY"]
        self.max_steps = params["MAX_STEPS"]
        self.epsilon_init = params["EPSILON_INIT"]
        self.epsilon_end = params["EPSILON_END"]
        self.epsilon_decay = params["EPSILON_DECAY"]

        self.gamma = params["GAMMA"]
        self.tau = params["TAU"]

    def choose_action(self, state, eps):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        sample = random.random()

        if sample > eps:
            with torch.no_grad():
                return int(self.policy_net(state).argmax())
        else:
            return random.choice(np.arange(self.n_actions))

    def eval_policy(self, eval_eps=100):
        scores = []
        for i_episode in range(eval_eps):
            score = 0
            state = self.env.reset()
            for i in range(self.max_steps):
                action = self.choose_action(state, 0)  # no random actions
                next_state, reward, done, _, _ = self.env.step(action)
                score += reward
                state = next_state
                if done:
                    break
            scores.append(score)
        success_rate = np.sum((np.array(scores) > 200)) / eval_eps
        print(f"Agent success rate: {success_rate}")
        return scores

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch

        q_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        q_target = rewards + self.gamma * q_next * (1 - dones)
        q_online = self.policy_net(states).gather(1, actions)

        loss = F.mse_loss(q_online, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, episodes, verbose=True, soft_update=False, end_when_solved=True):
        scores = []
        recent_scores = deque(maxlen=100)

        eps = self.epsilon_init
        for i_episode in range(1, episodes):
            state = self.env.reset()
            score = 0
            for i in range(self.max_steps):
                action = self.choose_action(state, eps)
                next_state, reward, done, _, _ = self.env.step(action)

                self.memory.append(state, action, reward, next_state, done)
                state = next_state
                score += reward

                if len(self.memory) > self.batch_size:
                    batch = self.memory.sample(self.batch_size)
                    self.update(batch)

                    if soft_update:
                        for p_param, t_param in zip(
                            self.policy_net.parameters(), self.target_net.parameters()
                        ):
                            t_param.data.copy_(
                                self.tau * p_param.data
                                + (1.0 - self.tau) * t_param.data
                            )

                if done:
                    break
            scores.append(score)
            recent_scores.append(score)

            eps = max(self.epsilon_end, eps * (1 - self.epsilon_decay))

            if (not soft_update) and self.target_update_delay != 0:
                if i_episode % self.target_update_delay == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            if np.all(np.array(recent_scores) >= 200) and end_when_solved:
                print(f"Reached score >= 200 in {i_episode} episodes.")
                break
            if verbose:
                print(
                    f"\rEpisode {i_episode} | Average score (last 100 ep): {np.mean(recent_scores):.2f} | sd: {np.std(recent_scores):.2f}",
                    end="",
                )
                # print('\rEpisode {} | Average score (last 100 ep): {:.2f}'.format(i_episode, np.mean(recent_scores)), end="")
                if i_episode % 100 == 0:
                    print(
                        f"\rEpisode {i_episode} | Average score (last 100 ep): {np.mean(recent_scores):.2f} | sd: {np.std(recent_scores):.2f}"
                    )
        return scores

    def save_model_checkpoint(self, fname="", fpath=SAVE_FPATH):
        ts = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
        torch.save(self.policy_net.state_dict(), fpath + fname + ts + ".pth")

    def make_video(self, fname=""):
        ts = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
        env = gym.make("LunarLander-v2")
        fpath = SAVE_FPATH + f"video/{fname}_" + ts + ".mp4"
        vid = video_recorder.VideoRecorder(env, path=fpath)
        state = env.reset()
        done = False
        score = 0
        while not done:
            frame = env.render(mode="rgb_array")
            vid.capture_frame()
            action = self.choose_action(state, 0)
            state, reward, done, _ = env.step(action)
            score += reward
        env.close()
        print(f"\nGenerated video of agent, score = {score}")
        return fpath
