# Nvidia Jetson

## General Info About Jetsons

[jtop](https://github.com/rbonghi/jetson_stats) is a nice tool similar to `htop` specialized for Jetson devices. You can refer to the repository for installation instructions.

### Clocks

Jetsons:

- use Boot and Power Management Processor (BPMP) for handling power management and clock management
- can print all the clock levels via `cat /sys/kernel/debug/clk/clk_summary`

### Nano

<https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-325/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/power_management_nano.html>

Power sysfs nodes: `/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/`  
Example: `sudo cat /sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input`

### TX2

<https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-325/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/power_management_tx2_32.html#>

Available power rails: GPU, SOC, WIFI, CPU, DDR, Main module

Supported modes and power efficiency:  
<https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-325/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/power_management_tx2_32.html#wwpID0E0DO0HA>

### Nano Orin

<https://docs.nvidia.com/jetson/archives/r35.4.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#>

Channel 1: VDD_IN

Total Module Power

Channel 2: VDD_CPU_GPU_CV

Total power consumed by CPU, CPU and CV cores i.e. DLA and PVA

Channel 3: VDD_SOC

Power consumed by SOC core which supplies to memory subsystem and various engines like nvdec, nvenc, vi, vic, isp etc.

### General Clock Levels (CPU, GPU, Memory)

`sudo /usr/bin/jetson_clocks --show` gives a nice visualization of the current clock levels.

---

BSP implements CPU Dynamic Frequency Scaling (DFS) with the Linux cpufreq subsystem. The cpufreq subsystem comprises:

- Platform drivers to implement the clock adjustment mechanism
- Governors to implement frequency scaling policies
- A core framework to connect governors to platform drivers

## Using Jetsons for CPU DVFS

In Jetson Nano, there is a single cluster of CPUs which includes both ARM and NVIDIA Denver cores. Denver is usually only for experimenting purposes so for a real workload, it makes sense to disable the Denver cores in order to get consistent performance regardless of the CPU scheduling.

In Jetson Nano Orin (jetson17), there are two CPU clusters with 4 CPUs and 2 CPUs respectively. All 6 CPUs are ARM.

It is important to note that the CPU frequency can only be set on a cluster-basis. This means that Jetson Nano can have up-to a single policy that determines the current frequency for all cores, whereas Jetson Nano Orin can have up-to two policies, one for each cluster.

All the interfacing with the frequency and power management is accomplished via device files. The following examples are correct for Jetson Nano Orin but other Jetson versions have very similar interfaces:

The prerequisite for setting a custom CPU frequency is to set the frequency governor to `userspace`. This can be done by writing to the file:

`/sys/devices/system/cpu/cpu%%X%%/cpufreq/scaling_governor`

To get the current cpu frequencies, you can read the file:

- `/sys/devices/system/cpu/cpu%%X%%/cpufreq/cpuinfo_cur_freq` for the actual frequency (numerical)
- `/sys/devices/system/cpu/cpu%%X%%/cpufreq/scaling_cur_freq` for the by policy set frequency (categorical from the possible values)

The available frequencies for a core is under:

- `/sys/devices/system/cpu/cpu%%X%%/cpufreq/scaling_available_frequencies`

In order to set the frequency of the CPU cores, you have to set both the minimum and maximum frequencies in:

- `/sys/devices/system/cpu/cpu%%X%%/cpufreq/scaling_min_freq`
`/sys/devices/system/cpu/cpu%%X%%/cpufreq/scaling_max_freq`

One thing to pay attention to is to set a single frequency value, you have to first set the maximum if you are trying to increase the frequency from the current, or minimum otherwise. This is because the reverse order causes the minimum to be larger than maximum for a brief time which is an invalid configuration.
