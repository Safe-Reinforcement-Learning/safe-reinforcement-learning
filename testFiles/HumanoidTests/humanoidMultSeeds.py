# Sam
# Testing base algorithms on humanoid on multiple seeds

from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train

totSteps = 10240000
perEpoch = 1024
wandBName = "Final Humanoid TRPO vs PPO Mult Seeds"
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
grid.add('train_cfgs:total_steps', [totSteps])
grid.add('algo_cfgs:steps_per_epoch', [perEpoch])
grid.add('seed', [30, 31, 32, 33, 34, 35, 36, 37, 38, 39])

grid.add('algo', base_policy)

grid.run(train, num_pool=20, parent_dir='resultsHumanoidMultSeeds')