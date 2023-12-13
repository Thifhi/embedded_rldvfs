TEMP_FILES = {
"CPU_TEMP" : "/sys/devices/virtual/thermal/thermal_zone0/temp",
"GPU_TEMP" : "/sys/devices/virtual/thermal/thermal_zone1/temp",
"SOC0_TEMP" : "/sys/devices/virtual/thermal/thermal_zone5/temp",
"SOC1_TEMP" : "/sys/devices/virtual/thermal/thermal_zone6/temp",
"SOC2_TEMP" : "/sys/devices/virtual/thermal/thermal_zone7/temp",
"TJ_TEMP" : "/sys/devices/virtual/thermal/thermal_zone8/temp",
}

class TemperatureReader:
    def __init__(self):
        self.open_files = {}
        for file_name, file in TEMP_FILES.items():
            self.open_files[file_name] = open(file, "r")
    
    def get_temperature(self):
        readings = {}
        for file_name, file in self.open_files.items():
            readings[file_name] = int(file.read().strip()) / 1000.0
            file.seek(0)
        return readings