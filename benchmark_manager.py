from pathlib import Path
import subprocess

PARSEC_BASE = Path(".") / "benchmarks" / "parsec" / "apps"
SUITES = {"parsec": PARSEC_BASE}

class BenchmarkManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.running_processes = []

    def run_benchmark(self, suite, benchmark):
        run_path = SUITES[suite] / benchmark / "run.sh"

        command = ["/bin/sh", run_path.resolve().as_posix()]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd = run_path.parent.resolve().as_posix())
        return process

    def run_benchmarks(self):
        for suite, benchmarks in self.cfg["benchmarks"].items():
            for benchmark in benchmarks:
                p = self.run_benchmark(suite, benchmark)
                self.running_processes.append((suite, benchmark, p))
    
    def poll_running(self):
        for suite, benchmark, p in self.running_processes:
            if p.poll() is None:
                return True
        return False
    
    def block_until_all_exit(self):
        for suite, benchmark, p in self.running_processes:
            stdout, stderr = p.communicate()
            if p.returncode == 0:
                print(f"Exited: {suite}-{benchmark}")
            else:
                print("Error in {suite}-{benchmark}:")
                print(stderr)