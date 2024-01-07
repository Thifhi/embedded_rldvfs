import os
import json
import pickle
import random
from collections import deque, namedtuple, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .rl_agent import RL_Agent
from . import rl_config as cfg

Experience = namedtuple("Experience", ["state", "action", "reward", "next_state"])

class ReplayBuffer:
    def __init__(self, batch_size, capacity):
        self.batch_size = batch_size
        self.buffer = deque(maxlen=capacity)

    def sample(self):
        experiences = random.sample(self.buffer, k=self.batch_size)
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([[e.action] for e in experiences])
        rewards = torch.FloatTensor([[e.reward] for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        return (states, actions, rewards, next_states)

    def add(self, state, action, reward, next_state):
        experience = Experience(state, action, reward, next_state)
        self.buffer.append(experience)

    def load(self, load_path):
        with open(load_path, "rb") as handle:
            self.buffer = pickle.load(handle)

    def save(self, save_path):
        with open(save_path, "wb") as handle:
            pickle.dump(self.buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def buffer_size(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, state_size, fc1_out_features, fc2_out_features, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, fc1_out_features)
        self.fc2 = nn.Linear(fc1_out_features, fc2_out_features)
        self.fc3 = nn.Linear(fc2_out_features, action_size)

    def forward(self, state):
        x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent(RL_Agent):
    def __init__(self, actions):
        self.config = cfg.DEFAULT_DEEP_Q_CONFIG
        self.running_stats = cfg.DEFAULT_RUNNING_STATS
        self.actions = actions
    
    def initialize_with_config(self):
        self.policy_net = DQN(len(self.config["STATE_ITEMS"]), self.config["FC1_OUT"], self.config["FC2_OUT"], len(self.actions))
        self.target_net = DQN(len(self.config["STATE_ITEMS"]), self.config["FC1_OUT"], self.config["FC2_OUT"], len(self.actions))
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config["LEARNING_RATE"])
        self.replay_buffer = ReplayBuffer(self.config["BATCH_SIZE"], self.config["REPLAY_BUFFER_CAPACITY"])

    def initialize_pretrained_with_config(self, pretrained_path):
        self.initialize_with_config()
        print("Loading pretrained model...")
        load_path = os.path.join(pretrained_path, cfg.POLICY_NET_PATH)
        self.policy_net.load_state_dict(torch.load(load_path))
        self.target_net.load_state_dict(torch.load(load_path))
        
    # Filter according to RL config 
    def filter_state(self, core_state, should_print_state=False):
        if core_state == None:
            return None
        ret = list(core_state.values())
        return ret

    def act(self, raw_state):
        dvfs = {}

        for core in [0, 4]:
            last_state = self.get_normalized_state_of_core(core)
            last_state = self.filter_state(last_state)
            last_action = self.get_action_of_core(core)
            
            self.update_state_of_core(core, raw_state)
            current_state = self.get_normalized_state_of_core(core)
            current_state = self.filter_state(current_state, True)
            # We calculate reward also in test to have a metric
            has_reward = False
            if last_action != None and last_state != None:
                reward = self.calculate_reward(core)
                has_reward = True
            if self.train:
                # If we are not acting on the core for the first time
                if has_reward:
                    self.replay_buffer.add(last_state, last_action, reward, current_state)
                # print("Buffer size: {0}".format(self.replay_buffer.buffer_size()))
                if self.replay_buffer.buffer_size() >= self.config["BATCH_SIZE"]:
                    self.learn()

            else:
                last_state = self.get_normalized_state_of_core(core)
                current_state = self.filter_state(last_state)

            action = self.get_action(current_state, epsilon_greedy=self.train)
            self.update_action_of_core(core, action)
            dvfs[core] = self.actions[action]
            # print("Chosen action: {0} MHz".format(dvfs[core]))

        return dvfs

    def calculate_reward(self, core_index):
        state = self.get_state_of_core(core_index)

        # Normalize temperature to constraint
        normalized_temperature = (state["CPU_TEMP"] - self.config["TEMPERATURE_CONSTRAINT"])

        reward_0 = self.config["K_Frequency"] * state[f"CPU-FREQ-{core_index}"]
        reward_1 = (-1) * self.config["K_Temperature"] * max(0, normalized_temperature)
        reward = reward_0 + reward_1

        # print("Reward... (from frequency: {:.3f}, from temperature: {:.3f})".format(reward_0, reward_1))
        # print("Final reward: {:.3f}".format(reward))

        self.running_stats["REWARDS"]["TRAIN" if self.train else "TEST"][self.running_stats["RUN_COUNTER"]].append(reward)

        return reward

    def learn(self):
        # Do this also for the first call to learn!
        if self.running_stats["LEARN_COUNTER"] % self.config["TARGET_UPDATE_FREQUENCY"] == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.running_stats["LEARN_COUNTER"] += 1

        states, actions, rewards, next_states = self.replay_buffer.sample()
        predicted_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            estimated_values = rewards + (self.config["GAMMA"] * self.target_net(next_states).detach().max(1)[0].unsqueeze(1))

        criterion = nn.MSELoss()
        loss = criterion(predicted_values, estimated_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.running_stats["Q_LOSSES"][self.running_stats["RUN_COUNTER"]].append(loss.item())

    
    def get_action(self, state, epsilon_greedy):
        effective_epsilon =  max(self.config["EPSILON_MIN"], self.config["EPSILON"] - (self.config["EPSILON_DECAY_PER_RUN"] * self.running_stats["RUN_COUNTER"]))
        # print("Effective epsilon: {0}".format(effective_epsilon))
        if epsilon_greedy and random.random() < effective_epsilon:
            # print("Choosing random action")
            action = random.randrange(0, len(self.actions))
        else:
            # print("Choosing best action")
            state = torch.FloatTensor(state)
            with torch.no_grad():
                prediction = self.policy_net(state)
            action = torch.argmax(prediction)
        return action
    
    def get_run_count(self):
        return self.running_stats["RUN_COUNTER"]

    def start_run(self, train):
        # print(self.running_stats["RUN_COUNTER"])
        if train:
            self.set_train()
            self.running_stats["REWARDS"]["TRAIN"][self.running_stats["RUN_COUNTER"]] = []
            self.running_stats["Q_LOSSES"][self.running_stats["RUN_COUNTER"]] = []
        else:
            self.set_test()
            self.running_stats["REWARDS"]["TEST"][self.running_stats["RUN_COUNTER"]] = []
    
    def finalize_run(self):
        self.running_stats["RUN_COUNTER"] += 1
    
    def load(self, path):
        with open(os.path.join(path, cfg.AGENT_CONFIG_PATH), "r") as handle:
            self.config = json.load(handle, object_pairs_hook=OrderedDict)
        with open(os.path.join(path, cfg.AGENT_RUNNING_STATS_PATH), "r") as handle:
            self.running_stats = json.load(handle, object_pairs_hook=OrderedDict)
        self.initialize_with_config()
        self.policy_net.load_state_dict(torch.load(os.path.join(path, cfg.POLICY_NET_PATH)))
        self.target_net.load_state_dict(torch.load(os.path.join(path, cfg.TARGET_NET_PATH)))
        self.replay_buffer.load(os.path.join(path, cfg.REPLAY_BUFFER_PATH))
    
    def save(self, path):
        torch.save(self.policy_net.state_dict(), os.path.join(path, cfg.POLICY_NET_PATH))
        torch.save(self.target_net.state_dict(), os.path.join(path, cfg.TARGET_NET_PATH))
        self.replay_buffer.save(os.path.join(path, cfg.REPLAY_BUFFER_PATH))
        with open(os.path.join(path, cfg.AGENT_CONFIG_PATH), "w") as handle:
            json.dump(self.config, handle, indent=2)
        with open(os.path.join(path, cfg.AGENT_RUNNING_STATS_PATH), "w") as handle:
            json.dump(self.running_stats, handle, indent=2)