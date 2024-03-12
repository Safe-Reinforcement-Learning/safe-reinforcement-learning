# Ian
# Increasing the steps per epoch and testing on multiple costs using PPOLag and TRPOLag, car. 

from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train
from multiprocessing import Process

totSteps = 10240000
perEpoch = 2048
wandBName = "Final Car Lagrangian Steps Per Epoch Mult Costs"
useWandB = True

# Setting up experiment grids
egLag1 = ExperimentGrid(exp_name='lag1fi')
egLag2 = ExperimentGrid(exp_name='lag1fj')
egLag3 = ExperimentGrid(exp_name='lag1fk')
egLag4 = ExperimentGrid(exp_name='lag1fl')
egLag5 = ExperimentGrid(exp_name='lag1fm')

# Set the algorithms.
lagNoSim = ['PPOLag',
               'TRPOLag']

# Set the environments.
m_envs = [
    'SafetyCarGoal2-v0'
]

lags = [egLag1, egLag2, egLag3, egLag4, egLag5]

for grid in lags:
    grid.add('env_id', m_envs)
    grid.add('logger_cfgs:use_wandb', [useWandB])
    grid.add('logger_cfgs:use_tensorboard', [True])
    grid.add('logger_cfgs:wandb_project', [wandBName])
    grid.add('train_cfgs:vector_env_nums', [1])
    grid.add('train_cfgs:torch_threads', [1])
    grid.add('train_cfgs:total_steps', [totSteps])
    grid.add('algo_cfgs:steps_per_epoch', [perEpoch])
    grid.add('seed', [20])

costs = [25, 50, 100, 150, 200]

for i in range(5):
    lags[i].add('algo', lagNoSim)
    lags[i].add('lagrange_cfgs:cost_limit', [costs[i]])

def experiment_grid_runner(grid, thunk, num_pool, parent_dir):
    grid.run(thunk, num_pool=num_pool, parent_dir=parent_dir)

processes = []
for i in range(5):
    p = Process(target=experiment_grid_runner, args=(lags[i], train, 2, f'expLagEpCost{i + 1}'))
    processes.append(p)
    p.start()

for p in processes:
   p.join()
