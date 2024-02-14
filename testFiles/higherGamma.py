# Sam 
# testing gamma of .999 on car, PPOLag and TRPOLag, cost limit 100. 

import omnisafe
from omnisafe.common.experiment_grid import ExperimentGrid
#from omnisafe.typing import NamedTuple, Tuple
from omnisafe.utils.exp_grid_tools import train
from multiprocessing import Process

totSteps = 10240000
perEpoch = 1024
wandBName = "Testing Higher Gamma"
useWandB = True

# Setting up experiment grids
lags = ExperimentGrid(exp_name='lags')
# Set the algorithms.
lagrangianAlgos = ['PPOLag',
               'TRPOLag']

# Set the environments.
m_envs = [
    'SafetyCarGoal2-v0'
]


lags.add('env_id', m_envs)
lags.add('logger_cfgs:use_wandb', [useWandB])
lags.add('logger_cfgs:use_tensorboard', [True])
lags.add('logger_cfgs:wandb_project', [wandBName])
lags.add('train_cfgs:vector_env_nums', [1])
lags.add('train_cfgs:torch_threads', [1])
lags.add('train_cfgs:total_steps', [totSteps])
lags.add('algo_cfgs:steps_per_epoch', [perEpoch])
lags.add('seed', [41])

lags.add('lagrange_cfgs:cost_limit', [100])
lags.add('algo_cfgs:gamma', [.999])
lags.add('algo', lagrangianAlgos)


lags.run(train, num_pool=2, parent_dir='carHigherGamma')
