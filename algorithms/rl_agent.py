from . import sim_config as sim_cfg

class RL_Agent():
    
    def initialize_state_action(self):
        self.last_state_per_core = [None] * sim_cfg.NUMBER_OF_CORES
        self.last_action_per_core = [None] * sim_cfg.NUMBER_OF_CORES

    def set_train(self):
        self.initialize_state_action()
        self.train = True
    
    def set_test(self):
        self.initialize_state_action()
        self.train = False 

    def guess_active_cores(self, raw_state):
        active_cores = []
        inactive_cores = []
        for core_index in range(sim_cfg.NUMBER_OF_CORES):
            if raw_state["cpi"][core_index] <= sim_cfg.CPI_THRESHOLD:
                active_cores.append(core_index)
            else:
                inactive_cores.append(core_index)
        return active_cores, inactive_cores

    def get_state_of_core(self, core_index):
        if self.last_state_per_core[core_index] is None:
            return None

        cpi, temperature, max_neighbor_temperature, average_temperature, frequency, power, mpki =  self.last_state_per_core[core_index]
        ret = {}
        ret["cpi"] = cpi
        ret["temperature"] = temperature
        ret["max_neighbor_temp"] = max_neighbor_temperature
        ret["average_neighbor_temp"] = average_temperature
        ret["frequency"] = frequency
        ret["power"] = power
        ret["mpki"] = mpki

        return ret
    
    def get_normalized_state_of_core(self, core_index):
        if self.last_state_per_core[core_index] is None:
            return None
        
        cpi, temperature, max_neighbor_temperature, average_temperature, frequency, power, mpki =  self.last_state_per_core[core_index]
        ret = {}
        # Normalize state
        ret["cpi"] = (cpi - sim_cfg.CPI_MEAN) * sim_cfg.CPI_NORMALIZER_COEFFICIENT
        ret["temperature"] = (temperature - sim_cfg.TEMPERATURE_MEAN) * sim_cfg.TEMPERATURE_NORMALIZER_COEFFICIENT
        ret["max_neighbor_temp"] = (max_neighbor_temperature - sim_cfg.TEMPERATURE_MEAN) * sim_cfg.TEMPERATURE_NORMALIZER_COEFFICIENT
        ret["average_neighbor_temp"] = (average_temperature - sim_cfg.TEMPERATURE_MEAN) * sim_cfg.TEMPERATURE_NORMALIZER_COEFFICIENT
        ret["frequency"] = (frequency - sim_cfg.FREQUENCY_MEAN) * sim_cfg.FREQUENCY_NORMALIZER_COEFFICIENT
        ret["power"] = (power - sim_cfg.POWER_MEAN) * sim_cfg.POWER_NORMALIZER_COEFFICIENT

        return ret

    def update_state_of_core(self, core_index, raw_state):
        cpi = raw_state["cpi"][core_index]
        temperature = raw_state["temperature"][core_index]
        max_neighbor_temperature = 0
        accumulated_temperature = 0
        neighbor_indices = RL_Agent.get_neighbor_indices(core_index)
        for core in neighbor_indices:
            max_neighbor_temperature = max(max_neighbor_temperature, raw_state["temperature"][core])
            accumulated_temperature += raw_state["temperature"][core]
        average_temperature = accumulated_temperature / len(neighbor_indices)
        frequency = raw_state["frequency"][core_index]
        power = raw_state["power"][core_index]
        mpki = raw_state["mpki"][core_index]
        self.last_state_per_core[core_index] = (cpi, temperature, max_neighbor_temperature, average_temperature, frequency, power, mpki)
    
    def get_action_of_core(self, core_index):
        return self.last_action_per_core[core_index]
    
    def update_action_of_core(self, core_index, action):
        self.last_action_per_core[core_index] = action

    def get_neighbor_indices(core_index):
        indices = []
        has_left = core_index % sim_cfg.PER_ROW_CORES != 0
        has_right = core_index % sim_cfg.PER_ROW_CORES != 7
        has_upper = core_index >= sim_cfg.PER_ROW_CORES
        has_lower = core_index < sim_cfg.NUMBER_OF_CORES - sim_cfg.PER_ROW_CORES

        if has_left:
            indices.append(core_index - 1)
            if has_upper:
                indices.append(core_index - 1 - sim_cfg.PER_ROW_CORES)
            if has_lower:
                indices.append(core_index - 1 + sim_cfg.PER_ROW_CORES)
        if has_right:
            indices.append(core_index + 1)
            if has_upper:
                indices.append(core_index + 1 - sim_cfg.PER_ROW_CORES)
            if has_lower:
                indices.append(core_index + 1 + sim_cfg.PER_ROW_CORES)
        if has_upper:
            indices.append(core_index - sim_cfg.PER_ROW_CORES)
        if has_lower:
            indices.append(core_index + sim_cfg.PER_ROW_CORES)

        return indices