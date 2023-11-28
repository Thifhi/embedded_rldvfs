# This was initially written with consideration that we can set the frequency per individual core. This is not the case
# in Jetson Nano TX2, so setting CPUS to [0] is sufficient.

GOVERNOR_NAME = "userspace"
GOVERNOR_PATH = "/sys/devices/system/cpu/cpu%%X%%/cpufreq/scaling_governor"

CPUS = [0, 4]
CPU_DVFS_ACTUAL_PATH = "/sys/devices/system/cpu/cpu%%X%%/cpufreq/cpuinfo_cur_freq"
CPU_DVFS_SCALING_PATH = "/sys/devices/system/cpu/cpu%%X%%/cpufreq/scaling_cur_freq"
CPU_DVFS_ENUM_PATH = "/sys/devices/system/cpu/cpu%%X%%/cpufreq/scaling_available_frequencies"
CPU_DVFS_LOWER_BOUND = "/sys/devices/system/cpu/cpu%%X%%/cpufreq/scaling_min_freq"
CPU_DVFS_UPPER_BOUND = "/sys/devices/system/cpu/cpu%%X%%/cpufreq/scaling_max_freq"

class DVFSController:
    def __init__(self):
        for cpu in CPUS:
            with open(GOVERNOR_PATH.replace("%%X%%", str(cpu)), "w") as f:
                f.write(GOVERNOR_NAME)
            with open(GOVERNOR_PATH.replace("%%X%%", str(cpu)), "r") as f:
                assert f.readline().strip() == GOVERNOR_NAME

        with open(CPU_DVFS_ENUM_PATH.replace("%%X%%", str(CPUS[0]))) as f:
            self.available_cpu_frequencies = [int(x) for x in f.readline().split()]
        
        for cpu in CPUS[1:]:
            with open(CPU_DVFS_ENUM_PATH.replace("%%X%%", str(cpu))) as f:
                assert self.available_cpu_frequencies == [int(x) for x in f.readline().split()]

        self.cpu_set_dvfs_files = {}
        for cpu in CPUS:
            f_lower = open(CPU_DVFS_LOWER_BOUND.replace("%%X%%", str(cpu)), "w", buffering=1)
            f_upper = open(CPU_DVFS_UPPER_BOUND.replace("%%X%%", str(cpu)), "w", buffering=1)
            self.cpu_set_dvfs_files[cpu] = (f_lower, f_upper)
            f_lower.write(str(self.available_cpu_frequencies[0]) + "\n")
            f_upper.write(str(self.available_cpu_frequencies[-1]) + "\n")

        self.cpu_read_dvfs_files = {}
        for cpu in CPUS:
            f = open(CPU_DVFS_ACTUAL_PATH.replace("%%X%%", str(cpu)), "r")
            self.cpu_read_dvfs_files[cpu] = f

        self.last_set_frequency = 0

    
    def read_cpu_frequency(self, cpu):
        self.cpu_read_dvfs_files[cpu].seek(0)
        return int(self.cpu_read_dvfs_files[cpu].readline().strip())
    
    def read_all_cpu_frequencies(self):
        all_freqs = []
        for cpu in CPUS:
            all_freqs.append(self.read_cpu_frequency(cpu))
        return all_freqs

    def set_all_cpu_frequencies(self, dvfs):
        for cpu in CPUS:
            self.set_cpu_frequency(cpu, dvfs)
    
    def set_cpu_frequency(self, cpu, dvfs):
        f_lower, f_upper = self.cpu_set_dvfs_files[cpu]
        if self.last_set_frequency < dvfs:
            f_upper.write(str(dvfs) + "\n")
            f_lower.write(str(dvfs) + "\n")
        else:
            f_lower.write(str(dvfs) + "\n")
            f_upper.write(str(dvfs) + "\n")
        self.last_set_frequency = dvfs


if __name__ == "__main__":
    import random
    import time
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    controller = DVFSController()
    counter = 0
    set_values = [0]
    actual_values = []

    EPOCH = 0.0001

    for i in range(50):
        start = time.time()
        set_freq = random.choice(controller.available_cpu_frequencies[3:])
        actual_freq = controller.read_all_cpu_frequencies()[0]
        controller.set_all_cpu_frequencies(set_freq)

        set_values.append(set_freq)
        actual_values.append(actual_freq)

        counter += 1
        end = time.time()
        print(end -  start)
        time.sleep(EPOCH)

    actual_values.append(0)

    index = np.arange(len(set_values))

    plt.plot(index, set_values, label='Set frequency')
    plt.plot(index, actual_values, label='Actual frequency')

    plt.legend()
    plt.savefig(f"{EPOCH}.png")