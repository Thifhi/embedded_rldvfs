import os
import yaml
from benchmark_manager import BenchmarkManager
import time
import threading
from algorithms.dqn import DQNAgent
from temperature_reader import TemperatureReader
from dvfs_controller import DVFSController

CONFIG_PATH = "config.yaml"

class PeriodicSleeper(threading.Thread):
    def __init__(self, task_function, period):
        super().__init__()
        self.task_function = task_function
        self.period = period
        self.i = 0
        self.t0 = time.time()
        self.start()

    def sleep(self):
        self.i += 1
        delta = self.t0 + self.period * self.i - time.time()
        if delta > 0:
            time.sleep(delta)
    
    def run(self):
        while True:
            if not self.task_function():
                break
            self.sleep()

def read_config():
    with open(CONFIG_PATH, 'r') as file:
        data = yaml.safe_load(file)
    return data

def read_state(dvfs_controller: DVFSController, temperature_reader: TemperatureReader):
    frequencies = dvfs_controller.read_all_cpu_frequencies()
    temperatures = temperature_reader.get_temperature()
    return {**frequencies, **temperatures}

def main():
    cfg = read_config()
    agent_folder = os.path.join("agents", "testing_agent")

    mgr = BenchmarkManager(cfg)
    
    dvfs_controller = DVFSController([0, 4])
    temperature_reader = TemperatureReader()
    
    agent = DQNAgent(dvfs_controller.available_cpu_frequencies)
    agent.initialize_with_config()

    def act_with_info():
        action = agent.act(read_state(dvfs_controller, temperature_reader))
        for cpu, dvfs in action.items():
            dvfs_controller.set_cpu_frequency(cpu, dvfs)
        return mgr.poll_running()

    for i in range(10):
        mgr.run_benchmarks()
        agent.start_run(True)
        run_count = agent.get_run_count()
        threads = PeriodicSleeper(act_with_info, 1e-2)
        threads.join()
        print(f"Exit run {i}...")
        agent.finalize_run()
        # Save everything after one run
        if run_count == 0:
            os.mkdir(agent_folder)
        agent.save(agent_folder)

if __name__ == "__main__":
    main()