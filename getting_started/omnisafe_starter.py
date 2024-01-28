import os
import sys

# To set up wandb login information, run `wandb login` or `wandb login --relogin` from your terminal.
# import wandb
# wandb.init(project="safe-rl", entity='safe-rl-comps')

import omnisafe
from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.typing import NamedTuple, Tuple

def train(
    exp_id: str, algo: str, env_id: str, custom_cfgs: NamedTuple
) -> Tuple[float, float, float]:
    """Train a policy from exp-x config with OmniSafe.

    Args:
        exp_id (str): Experiment ID.
        algo (str): Algorithm to train.
        env_id (str): The name of test environment.
        custom_cfgs (NamedTuple): Custom configurations.
        num_threads (int, optional): Number of threads. Defaults to 6.
    """
    terminal_log_name = 'terminal.log'
    error_log_name = 'error.log'
    if 'seed' in custom_cfgs:
        terminal_log_name = f'seed{custom_cfgs["seed"]}_{terminal_log_name}'
        error_log_name = f'seed{custom_cfgs["seed"]}_{error_log_name}'
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f'exp-x: {exp_id} is training...')
    if not os.path.exists(custom_cfgs['logger_cfgs']['log_dir']):
        os.makedirs(custom_cfgs['logger_cfgs']['log_dir'], exist_ok=True)
    # pylint: disable-next=consider-using-with
    sys.stdout = open(
        os.path.join(f'{custom_cfgs["logger_cfgs"]["log_dir"]}', terminal_log_name),
        'w',
        encoding='utf-8',
    )
    # pylint: disable-next=consider-using-with
    sys.stderr = open(
        os.path.join(f'{custom_cfgs["logger_cfgs"]["log_dir"]}', error_log_name),
        'w',
        encoding='utf-8',
    )
    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    reward, cost, ep_len = agent.learn()
    return reward, cost, ep_len

eg = ExperimentGrid(exp_name='Tutorial_benchmark')
# Set the algorithms.
base_policy = ['PPO',
               'PPOLag',
               'PPOSimmerPID',
               'PPOLagSimmerPID',
               'TRPO',
               'TRPOLag',
               'TRPOSimmerPID',
               'TRPOLagSimmerPID']

# Set the environments.
m_envs = [
    'SafetyCarGoal2-v0',
    'SafetyHumanoidVelocity-v1'
]
eg.add('env_id', m_envs)
eg.add('algo', base_policy)
eg.add('logger_cfgs:use_wandb', [True])
eg.add('logger_cfgs:use_tensorboard', [True])
eg.add('logger_cfgs:wandb_project', ['the_wandb_project_name'])
eg.add('train_cfgs:vector_env_nums', [1])
eg.add('train_cfgs:torch_threads', [1])
eg.add('train_cfgs:total_steps', [2048])
eg.add('algo_cfgs:steps_per_epoch', [1024])
eg.add('seed', [0])

eg.run(train, num_pool=16)
eg.analyze(parameter='algo', values=base_policy, compare_num=None, cost_limit=None)

eg.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
eg.evaluate(num_episodes=1)