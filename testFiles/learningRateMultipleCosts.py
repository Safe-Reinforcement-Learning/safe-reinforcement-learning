# Sam 
# Trying a lower lagrange learning rate on multiple costs, car. 

import omnisafe
from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train

totSteps = 10240000
perEpoch = 1024
wandBName = "10000 Epoch Car Lagrange Learning Rate on Multiple Costs"
useWandB = True

# Setting up experiment grids
grid = ExperimentGrid(exp_name='lagLRMultCosts')

# Set the algorithms.
lagrangian = ['PPOLag',
               'TRPOLag']

# Set the environments.
m_envs = [
    'SafetyCarGoal2-v0'
]


grid.add('env_id', m_envs)
grid.add('logger_cfgs:use_wandb', [useWandB])
grid.add('logger_cfgs:use_tensorboard', [True])
grid.add('logger_cfgs:wandb_project', [wandBName])
grid.add('train_cfgs:vector_env_nums', [1])
grid.add('train_cfgs:torch_threads', [1])
grid.add('train_cfgs:total_steps', [totSteps])
grid.add('algo_cfgs:steps_per_epoch', [perEpoch])
grid.add('seed', [10])
grid.add('lagrange_cfgs:cost_limit', [25, 50, 100, 150, 175])
grid.add('lagrange_cfgs:lambda_lr', [.00035, .035])

grid.add('algo', lagrangian)

grid.run(train, num_pool=20, parent_dir='LRMultCosts')