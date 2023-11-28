def main():
    freq_low = 114750000
    freq_high = 1300500000
    path = "/sys/devices/gpu.0/devfreq/17000000.gp10b/cur_freq"
    with open(path, "w") as f:
        f.write(str(freq_low))

if __name__ == "__main__":
    main()