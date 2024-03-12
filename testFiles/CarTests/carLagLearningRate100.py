# Sam 
# Trying lower lagrange learning rates on budget 100, car. 

from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train

totSteps = 10240000
perEpoch = 1024
wandBName = "Final Car Lagrange Learning Rates on Budget 100"
useWandB = True

# Setting up experiment grids
grid = ExperimentGrid(exp_name='lagLR')

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
grid.add('seed', [20])
grid.add('lagrange_cfgs:cost_limit', [100])
grid.add('lagrange_cfgs:lambda_lr', [.035, .0035, .00035])

grid.add('algo', lagrangian)


grid.run(train, num_pool=6, parent_dir='resultsCarLagLearningRate100')