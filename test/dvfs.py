from enum import Enum
import subprocess
import pathlib
from typing import Union


class Device(str, Enum):
    CPU = "cpu"
    GPU = "gpu"

class CPUGovernor(str, Enum):
    INTERACTIVE = "interactive"
    CONSERVATIVE = "conservative"
    ONDEMAND = "ondemand"
    USERSPACE = "userspace"
    POWERSAVE = "powersave"
    PERFORMANCE = "performance"
    SCHEDUTIL = "schedutil"

class GPUGovernor(str, Enum):
    WMARK_ACTIVE = "wmark_active"
    WMARK_SIMPLE = "wmark_simple"
    NVHOST_PODGOV = "nvhost_podgov"
    USERSPACE = "userspace"
    PERFORMANCE = "performance"
    SIMPLE_ONDEMAND = "simple_ondemand"

FREQUENCIES = {
    # TODO fill for each device type
    Device.CPU: [518400, 614400, 710400, 825600, 921600, 1036800, 1132800, 1224000, 1326000, 1428000, 1479000],
    Device.GPU: [76800000, 153600000, 230400000, 307200000, 384000000, 460800000,
                 537600000, 614400000, 691200000, 768000000, 844800000, 921600000]
}

def get_frequency(device: Device) -> int:
    path = "/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq" if device is Device.CPU else "/sys/devices/57000000.gpu/devfreq/57000000.gpu/cur_freq"
    with open(path) as f:
        return int(f.read())

def set_frequency(device: Device, frequency: int) -> None:
    current_governor = get_governor(device)
    if current_governor != "userspace":
        raise AssertionError(f"The {device} governor should be 'userspace' to be able to set frequency, but the current governor is {current_governor}")

    if frequency not in FREQUENCIES[Device.CPU]:
        raise ValueError("The frequency {frequency} is not available for {device}.")

    subprocess.call(
        [
            "/bin/sh",
            f"{str(pathlib.Path(__file__).parent.resolve())}/{device}/set_frequency.sh",
            str(frequency)
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    assert get_frequency(device) == frequency

def get_governor(device: Device) -> Union[CPUGovernor, GPUGovernor]:
    path = "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor" if device is Device.CPU else "/sys/devices/57000000.gpu/devfreq/57000000.gpu/governor"
    with open(path) as f:
        return f.read().rstrip()

def set_cpu_governor(governor: CPUGovernor) -> None:
    subprocess.call(
        [
            "/bin/sh",
            f"{str(pathlib.Path(__file__).parent.resolve())}/cpu/set_governor.sh",
            governor
        ]
    )

    assert get_governor(Device.CPU) == governor

def set_gpu_governor(governor: GPUGovernor) -> None:
    subprocess.call(
        [
            "/bin/sh",
            f"{str(pathlib.Path(__file__).parent.resolve())}/gpu/set_governor.sh",
            governor
        ]        
    )

    assert get_governor(Device.GPU) == governor
