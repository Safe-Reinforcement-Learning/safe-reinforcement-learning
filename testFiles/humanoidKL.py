# Ian
# Testing different KL-early stop and KL targets on humanoid using PPO and TRPO

from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train

totSteps = 10240000
perEpoch = 1024
wandBName = "Some KL Experiments 2 point O"
useWandB = True

# Setting up experiment grids
grid1 = ExperimentGrid(exp_name='base')
grid2 = ExperimentGrid(exp_name='base1')

# Set the algorithms.
base_policy = ['PPO',
               'TRPO']

# Set the environments.
m_envs = [
    'SafetyHumanoidVelocity-v1'
]

grids = [grid1, grid2]

for grid in grids:
    grid.add('env_id', m_envs)
    grid.add('logger_cfgs:use_wandb', [useWandB])
    grid.add('logger_cfgs:use_tensorboard', [True])
    grid.add('logger_cfgs:wandb_project', [wandBName])
    grid.add('train_cfgs:vector_env_nums', [1])
    grid.add('train_cfgs:torch_threads', [1])
    grid.add('train_cfgs:total_steps', [totSteps])
    grid.add('algo_cfgs:steps_per_epoch', [perEpoch])
    grid.add('seed', [1])

grid1.add('algo', base_policy)
grid1.add('algo_cfgs:target_kl', [.2, .02, .002])
grid1.add('algo_cfgs:kl_early_stop',[True])

grid2.add('algo', base_policy)
grid2.add('algo_cfgs:kl_early_stop',[False])


grid1.run(train, num_pool=6, parent_dir='expKL')
grid2.run(train, num_pool=2, parent_dir='expKLB')

