# Ian
# Increasing the steps per epoch and testing on multiple costs using PPOLag and TRPOLag, car. 

import omnisafe
from omnisafe.common.experiment_grid import ExperimentGrid
#from omnisafe.typing import NamedTuple, Tuple
from omnisafe.utils.exp_grid_tools import train
from multiprocessing import Process

# Final Car Lagrangian Steps Per Epoch Mult Costs
# Rerun the different steps per epoch with costs 25, 50, 100, 150, 200

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

# egLagSim1 = ExperimentGrid(exp_name='lagSim1')
# egLagSim2 = ExperimentGrid(exp_name='lagSim2')
# egLagSim3 = ExperimentGrid(exp_name='lagSim3')
# egLagSim4 = ExperimentGrid(exp_name='lagSim4')
# egLagSim5 = ExperimentGrid(exp_name='lagSim5')

# egBase = ExperimentGrid(exp_name='base')

# egSim1 = ExperimentGrid(exp_name='s1')
# egSim2 = ExperimentGrid(exp_name='s2')
# egSim3 = ExperimentGrid(exp_name='s3')
# egSim4 = ExperimentGrid(exp_name='s4')
# egSim5 = ExperimentGrid(exp_name='s5')

# Set the algorithms.
# lagSim = ['PPOLagSimmerPID',
#             'TRPOLagSimmerPID']
# base_policy = ['PPO',
#                'TRPO']
lagNoSim = ['PPOLag',
               'TRPOLag']
# sim = ['PPOSimmerPID',
#                'TRPOSimmerPID']

# Set the environments.
m_envs = [
    'SafetyCarGoal2-v0'
]
# SafetyCarGoal2-v0	
# 'SafetyCarGoal1-v0'

lags = [egLag1, egLag2, egLag3, egLag4, egLag5]

# lagSims = [egLagSim1, egLagSim2, egLagSim3, egLagSim4, egLagSim5]

# sims = [egSim1, egSim2, egSim3, egSim4, egSim5]

# grids = (egLag1, egLag2, egLag3, egLag4, egLag5, egLagSim1, 
#          egLagSim2, egLagSim3, egLagSim4, egLagSim5, egSim1, 
#          egSim2, egSim3, egSim4, egSim5, egBase)

#for grid in grids: 
for grid in lags:
    grid.add('env_id', m_envs)
    grid.add('logger_cfgs:use_wandb', [useWandB])
    grid.add('logger_cfgs:use_tensorboard', [True])
    grid.add('logger_cfgs:wandb_project', [wandBName])
    grid.add('train_cfgs:vector_env_nums', [1])
    grid.add('train_cfgs:torch_threads', [1])
    grid.add('train_cfgs:total_steps', [totSteps])
    grid.add('algo_cfgs:steps_per_epoch', [perEpoch])
    grid.add('seed', [1])


costs = [25, 50, 100, 150, 200]

for i in range(5):
    # lagSims[i].add('algo', lagSim)
    # lagSims[i].add('algo_cfgs:safety_budget', [costs[i]])
    # lagSims[i].add('algo_cfgs:upper_budget', [costs[i]])
    # lagSims[i].add('lagrange_cfgs:cost_limit', [costs[i]])


    lags[i].add('algo', lagNoSim)
    lags[i].add('lagrange_cfgs:cost_limit', [costs[i]])

    # sims[i].add('algo', sim)
    # sims[i].add('algo_cfgs:safety_budget', [costs[i]])
    # sims[i].add('algo_cfgs:upper_budget', [costs[i]])


# egBase.add('algo', base_policy)
it = 1
#for grid in grids: 

# for grid in sims: 
#     it+=1
#     grid.run(train, num_pool =20, parent_dir='expSimmer'+str(it))

def experiment_grid_runner(grid, thunk, num_pool, parent_dir):
    grid.run(thunk, num_pool=num_pool, parent_dir=parent_dir)

processes = []
for i in range(5):
    p = Process(target=experiment_grid_runner, args=(lags[i], train, 4, f'expLagEpCost{i + 1}'))
    processes.append(p)
    p.start()

for p in processes:
   p.join()
