# Ian
# Running safety algorithms with 20000 steps per epoch, multiple seeds at budget 100

from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train
from multiprocessing import Process

### RUN ALL SAFETY ALGORITHMS WITH MULTIPLE SEEDS COST 100

totSteps = 10000000
perEpoch = 20000
wandBName = "Final Car Comparison, Budget 100, 20000 Epochs"
useWandB = True

# Setting up experiment grids
Lag1 = ExperimentGrid(exp_name='AllLag2')

Base = ExperimentGrid(exp_name='AllBase2')

Sim1 = ExperimentGrid(exp_name='AllSim2')

# Set the algorithms.
base_policy = ['PPO',
               'TRPO']
lagNoSim = ['PPOLag',
               'TRPOLag']
sim = ['PPOSimmerPID',
               'TRPOSimmerPID']

# Set the environments.
m_envs = [
    'SafetyCarGoal2-v0'
]

grids = (Lag1, Sim1, Base)

#for grid in grids: 
for grid in grids:
    grid.add('env_id', m_envs)
    grid.add('logger_cfgs:use_wandb', [useWandB])
    grid.add('logger_cfgs:use_tensorboard', [True])
    grid.add('logger_cfgs:wandb_project', [wandBName])
    grid.add('train_cfgs:vector_env_nums', [1])
    grid.add('train_cfgs:torch_threads', [1])
    grid.add('train_cfgs:total_steps', [totSteps])
    grid.add('algo_cfgs:steps_per_epoch', [perEpoch])
    grid.add('seed', [78, 79])
    # grid.add('seed', [76, 77])

cost = [100]

Lag1.add('algo', lagNoSim)
Lag1.add('lagrange_cfgs:cost_limit', cost)

Sim1.add('algo', sim)
Sim1.add('algo_cfgs:safety_budget', cost)
Sim1.add('algo_cfgs:upper_budget', cost)

Base.add('algo', base_policy)

def experiment_grid_runner(grid, thunk, num_pool, parent_dir):
    grid.run(thunk, num_pool=num_pool, parent_dir=parent_dir)

processes = []
for i in range(3):
    p = Process(target=experiment_grid_runner, args=(grids[i], train, 4, f'carCost20000{i + 1}'))
    processes.append(p)
    p.start()

for p in processes:
   p.join()
