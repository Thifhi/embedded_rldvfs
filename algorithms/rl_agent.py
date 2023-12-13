class RL_Agent():
    
    def initialize_state_action(self):
        self.last_state_per_core = {}
        self.last_action_per_core = {}

    def set_train(self):
        self.initialize_state_action()
        self.train = True
    
    def set_test(self):
        self.initialize_state_action()
        self.train = False 

    def get_state_of_core(self, core_index):
        if not core_index in self.last_state_per_core:
            return None

        return self.last_state_per_core[core_index]
    
    # def get_normalized_state_of_core(self, core_index):
    #     if self.last_state_per_core[core_index] is None:
    #         return None
        
    #     cpi, temperature, max_neighbor_temperature, average_temperature, frequency, power, mpki =  self.last_state_per_core[core_index]
    #     ret = {}
    #     # Normalize state
    #     ret["cpi"] = (cpi - sim_cfg.CPI_MEAN) * sim_cfg.CPI_NORMALIZER_COEFFICIENT
    #     ret["temperature"] = (temperature - sim_cfg.TEMPERATURE_MEAN) * sim_cfg.TEMPERATURE_NORMALIZER_COEFFICIENT
    #     ret["max_neighbor_temp"] = (max_neighbor_temperature - sim_cfg.TEMPERATURE_MEAN) * sim_cfg.TEMPERATURE_NORMALIZER_COEFFICIENT
    #     ret["average_neighbor_temp"] = (average_temperature - sim_cfg.TEMPERATURE_MEAN) * sim_cfg.TEMPERATURE_NORMALIZER_COEFFICIENT
    #     ret["frequency"] = (frequency - sim_cfg.FREQUENCY_MEAN) * sim_cfg.FREQUENCY_NORMALIZER_COEFFICIENT
    #     ret["power"] = (power - sim_cfg.POWER_MEAN) * sim_cfg.POWER_NORMALIZER_COEFFICIENT

    #     return ret

    def update_state_of_core(self, core_index, raw_state):
        self.last_state_per_core[core_index] = raw_state
    
    def get_action_of_core(self, core_index):
        if not core_index in self.last_action_per_core:
            return None

        return self.last_action_per_core[core_index]
    
    def update_action_of_core(self, core_index, action):
        self.last_action_per_core[core_index] = action
