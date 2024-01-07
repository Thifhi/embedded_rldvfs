# Using RL DVFS

## General

The project is structured in such a way that the simulation framework, HotSniper is completely separated from the 
machine learning codebase that resides under `rl`.

## RL
The training process is initiated in the `trainer.py` file. Once it is called, it will spawn build the agent object according to the configuration under `rl_config.py`. `rl_config.py` file contains various variables such as the algorithm that will be used for training, the saving paths of the trained models, the hyperparameters of the algorithm (epsilon, gamma, decay values) and reward coefficients. If an agent with the same name already exists, the trainer will load that agent instead and continue the training process from that point on.

After the agent object is constructed, the simulation is started with the benchmarks that are stated under `sim_config.py`. Similar to `rl_config.py`, sim config contains various variables that will determine the sim behavior. It also contains constants such as normalizer coefficients of the state components. The variables under "Base Config" are Sniper config variables that will configure Sniper correctly before each run.

The simulation and agent interaction is handled in the RL side by the `simulation_control.py`. It begins by reading from a named pipe and blocks until a state information is returned from the simulation. It receives this information every simulation epoch and passes it down to the agent by calling its `act` method. The agent calculates the DVFS values and returns as the return value from `act`, which are written to another named pipe by the simulation control, upon which it blocks until the next state arrives.

## Sniper
The simulation is initiated from the `simulationcontrol/run.py`. The setup allows running multiple simulations simultaneously by setting the training directory a temporary folder for all requried components. After each training run, the important results and logs are copied to their correct place. This way, two running simulations can not influence each other and the user can run as many simulations as needed at the same time. The actual temporary folder location is passed down from the RL module so that RL module will also have access to the directory and can perform the necessary copy operations.

All simulation configuration values that are relevant reside under `config/base.cfg`. Here, the user can define new configuration values that can be set by its name with the convention:
- key: value # cfg:name

If the name is set in the `sim_config.py` from RL, the line will be left uncommented and its value will be used for the training.

The communication with the named pipe from the Sniper side is handled by the `scripts/dvfs_rl.py` file. Any new state information can be passed to the RL module by modifying this file.