import yaml
from benchmark_manager import BenchmarkManager

CONFIG_PATH = "config.yaml"

def read_config():
    with open(CONFIG_PATH, 'r') as file:
        data = yaml.safe_load(file)
    return data

def main():
    cfg = read_config()
    mgr = BenchmarkManager(cfg)
    mgr.run_benchmarks()
    mgr.block_until_all_exit()

if __name__ == "__main__":
    main()