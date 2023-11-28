from .rl_agent import RL_Agent
from . import rl_config as cfg
import numpy as np
import random
import os
from collections import OrderedDict
import json

class TableQAgent(RL_Agent):
    def __init__(self):
        self.config = cfg.DEFAULT_TABLE_Q_CONFIG
        self.running_stats = cfg.DEFAULT_RUNNING_STATS
    
    def initialize_with_config(self):
        ips_step_size = len(self.config["CPI_STEPS"])
        temperature_step_size = len(self.config["TEMPERATURE_STEPS"])
        freq_step_size = len(self.config["FREQUENCY_STEPS"])
        power_step_size = len(self.config["POWER_STEPS"])
        action_step_size = len(self.config["ACTIONS"])
        self.Q = np.zeros((ips_step_size, temperature_step_size, temperature_step_size, temperature_step_size, freq_step_size, power_step_size, action_step_size))
    
    def initialize_pretrained_with_config(self, pretrained_path):
        print("Loading pretrained model...")
        self.Q = np.load(os.path.join(pretrained_path, cfg.Q_TABLE_PATH))

    def act(self, raw_state):
        active_cores, inactive_cores = self.guess_active_cores(raw_state)

        dvfs = {}

        for core in inactive_cores:
            # Lowest freq
            dvfs[str(core)] = self.config["ACTIONS"][0]

        for core in active_cores:
            last_state = self.filter_state(self.get_state_of_core(core))
            last_action = self.get_action_of_core(core)
            self.update_state_of_core(core, raw_state)
            current_state = self.filter_state(self.get_state_of_core(core), should_print_state=True)
            # We calculate reward also in test to have a metric
            has_reward = False
            if last_action != None and last_state != None:
                reward = self.calculate_reward(core)
                has_reward = True
            if self.train:
                # If we are not acting on the core for the first time
                if has_reward:
                    self.learn(last_state, current_state, last_action, reward)

            else:
                self.update_state_of_core(core, raw_state)
                current_state = self.filter_state(self.get_state_of_core(core))

            action = self.get_action(current_state, epsilon_greedy=self.train)
            self.update_action_of_core(core, action)
            dvfs[str(core)] = self.config["ACTIONS"][action]
            print("Chosen action: {0} MHz".format(dvfs[str(core)]))

        return dvfs

    def calculate_reward(self, core_index):
        state = self.get_state_of_core(core_index)

        # Normalize temperature to constraint
        normalized_temperature = (state["temperature"] - self.config["TEMPERATURE_CONSTRAINT"])

        reward_0 = self.config["K_Frequency"] * state["frequency"]
        reward_1 = (-1) * self.config["K_Temperature"] * max(0, normalized_temperature)
        reward = reward_0 + reward_1

        print("Reward... (from frequency: {:.3f}, from temperature: {:.3f})".format(reward_0, reward_1))
        print("Final reward: {:.3f}".format(reward))

        self.running_stats["REWARDS"]["TRAIN" if self.train else "TEST"][self.running_stats["RUN_COUNTER"]].append(reward)

        return reward

    def learn(self, last_state, current_state, last_action, reward):
        self.running_stats["LEARN_COUNTER"] += 1

        last_state_indices = self.state_to_indices(last_state)
        last_state_action_indices = tuple(last_state_indices) + (last_action,)
        current_state_indices = self.state_to_indices(current_state)
        print("Current state indices: {}".format(current_state_indices))
        max_current_Q_value = np.max(self.Q[current_state_indices])
        last_Q_value = self.Q[last_state_action_indices]
        Q_difference = reward + self.config["GAMMA"] * max_current_Q_value - last_Q_value
        loss = Q_difference ** 2
        self.Q[last_state_action_indices] += self.config["LEARNING_RATE"] * Q_difference

        self.running_stats["Q_LOSSES"][self.running_stats["RUN_COUNTER"]].append(loss)

    
    def get_action(self, state, epsilon_greedy):
        effective_epsilon =  max(self.config["EPSILON_MIN"], self.config["EPSILON"] - (self.config["EPSILON_DECAY_PER_RUN"] * self.running_stats["RUN_COUNTER"]))
        print("Effective epsilon: {0}".format(effective_epsilon))
        if epsilon_greedy and random.random() < effective_epsilon:
            print("Choosing random action")
            action = random.randrange(0, len(self.config["ACTIONS"]))
        else:
            print("Choosing best action")
            q_values = self.Q[tuple(self.state_to_indices(state))]
            action = q_values.argmax()
        return action
    
    def state_to_indices(self, state):
        indices = []

        cpi_index = 0
        if "cpi" in state:
            for i in range(len(self.config["CPI_STEPS"])):
                if state["cpi"] >= self.config["CPI_STEPS"][i]:
                    cpi_index = i
                else:
                    break
            indices.append(cpi_index)

        for temperature_state in ("temperature", "max_neighbor_temp", "average_neighbor_temp"):
            if temperature_state in state:
                temperature_index = 0
                for i in range(len(self.config["TEMPERATURE_STEPS"])):
                    if state[temperature_state] >= self.config["TEMPERATURE_STEPS"][i]:
                        temperature_index = i
                    else:
                        break
                indices.append(temperature_index)

        frequency_index = 0
        if "frequency" in state:
            for i in range(len(self.config["FREQUENCY_STEPS"])):
                if state["frequency"] >= self.config["FREQUENCY_STEPS"][i]:
                    frequency_index = i
                else:
                    break
            indices.append(frequency_index)

        power_index = 0
        if "power" in state:
            for i in range(len(self.config["POWER_STEPS"])):
                if state["power"] >= self.config["POWER_STEPS"][i]:
                    power_index = i
                else:
                    break
            indices.append(power_index)
        return tuple(indices)
            
    def filter_state(self, core_state, should_print_state=False):
        if core_state == None:
            return None
        if (should_print_state):
            print_state = "Current core state: "
            for item in self.config["STATE_ITEMS"]:
                print_state += item + ": " + "{:.3f}".format(core_state[item]) + ", "
            print(print_state[:-2])
        return {item: core_state[item] for item in self.config["STATE_ITEMS"]}

    def get_run_count(self):
        return self.running_stats["RUN_COUNTER"]

    def start_run(self, train):
        if train:
            self.set_train()
            self.running_stats["Q_LOSSES"][self.running_stats["RUN_COUNTER"]] = []
            self.running_stats["REWARDS"]["TRAIN"][self.running_stats["RUN_COUNTER"]] = []
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
        self.Q = np.load(os.path.join(path, cfg.Q_TABLE_PATH))
    
    def save(self, path):
        with open(os.path.join(path, cfg.AGENT_CONFIG_PATH), "w") as handle:
            json.dump(self.config, handle, indent=2)
        with open(os.path.join(path, cfg.AGENT_RUNNING_STATS_PATH), "w") as handle:
            json.dump(self.running_stats, handle, indent=2)
        np.save(os.path.join(path, cfg.Q_TABLE_PATH), self.Q)