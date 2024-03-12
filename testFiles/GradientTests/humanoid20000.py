# Ian 
# Humanoid with 20000 steps per epoch and 5000 epochs (10x total steps from
# what our other experiments have been), base algos with 5 seeds

from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train

wandBName = "Final Humanoid TRPO vs PPO 20000 Steps per Epoch"
useWandB = True

# Setting up experiment grids
grid = ExperimentGrid(exp_name='base')

# Set the algorithms.
base_policy = ['PPO',
               'TRPO']

# Set the environments.
m_envs = [
    'SafetyHumanoidVelocity-v1'
]

grid.add('env_id', m_envs)
grid.add('logger_cfgs:use_wandb', [useWandB])
grid.add('logger_cfgs:use_tensorboard', [True])
grid.add('logger_cfgs:wandb_project', [wandBName])
grid.add('train_cfgs:vector_env_nums', [1])
grid.add('train_cfgs:torch_threads', [1])
grid.add('algo_cfgs:steps_per_epoch', [20000])
grid.add('train_cfgs:total_steps', [20000 * 5000])
grid.add('seed', [1, 5, 10, 15, 20])

grid.add('algo', base_policy)

grid.run(train, num_pool=10, parent_dir='5000EpochsTest')
