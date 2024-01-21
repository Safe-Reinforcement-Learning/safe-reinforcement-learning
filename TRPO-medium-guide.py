"""
Algo Team 1: TRPO
Lily and Emma
this is from 
https://medium.com/@vladogim97/trpo-minimal-pytorch-implementation-859e46c4232e
 this medium article

 we are using it because we are lost
 most comments are our own 

 Will either recreate this later with more information or heavily cite this guy
"""


import gym
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.distributions import Categorical
from collections import namedtuple
from torch.optim import Adam

env = gym.make('CartPole-v0')

state_size = env.observation_space.shape[0]
num_actions = env.action_space.n

#vine TRPO
#multiple rollouts to create traj
Rollout = namedtuple('Rollout',
                     ['states', 'actions', 'rewards', 'next_states', ])

#10 trajectory vine paths
def train(epochs=100, num_rollouts=10, render_frequency = None):
    mean_total_rewards = []
    global_rollout = 0

    for epoch in range(epochs):
        rollouts = []
        rollout_total_rewards = []

        for t in range(num_rollouts):
            state = env.reset() #vine path we need reset the former vine path
            done = False #haven't done it yet

            samples = [] #collecting

            while not done:
                if render_frequency is not None and global_rollout % render_frequency == 0:
                    env.render()
                with torch.no_grad(): #gradient?
                    action = get_action(state) 
                next_state, reward, done, _ = env.step(action) #take an action
                #this determines if we're done

                # Collect samples
                samples.append((state, action, reward, next_state))

                state = next_state

            # Transpose our samples
            states, actions, rewards, next_states = zip(*samples)
            
            #i'm building a set of tensors
            states = torch.stack([torch.from_numpy(state) for state in states], dim=0).float()
            #also a tensor
            next_states = torch.stack([torch.from_numpy(state) for state in next_states], dim=0).float()
            
            #becomes 1 tensor
            actions = torch.as_tensor(actions).unsqueeze(1)
            rewards = torch.as_tensor(rewards).unsqueeze(1)
            
            #adding the new rollout to our set of rollouts
            rollouts.append(Rollout(states, actions, rewards, next_states))

            rollout_total_rewards.append(rewards.sum().item())
            global_rollout += 1
        
        update_agent(rollouts)
        mtr = np.mean(rollout_total_rewards)
        print(f'E: {epoch}.\tMean total reward across {num_rollouts} rollouts: {mtr}')

        mean_total_rewards.append(mtr)
    
    plt.plot(mean_total_rewards)
    plt.show()


####### ACTOR ########
actor_hidden = 32 #arbitrary????? IDK

#neural network
#will output probabilities for the actions in the current state
actor = nn.Sequential(nn.Linear(state_size, actor_hidden),
                      nn.ReLU(),
                      nn.Linear(actor_hidden, num_actions),
                      nn.Softmax(dim=1))

#to get our action
def get_action(state):
    #what is a batch? squeeze? IDK
    state = torch.tensor(state).float().unsqueeze(0)  # Turn state into a batch with a single element
    dist = Categorical(actor(state))  # Create a distribution from probabilities for actions
    return dist.sample().item() #returns an action from our distribution 

#Using a critic rather than KL or Lagrangian measures of difference 
# Critic takes a state and returns its values
critic_hidden = 32 #prob arbitrary

#nn
critic = nn.Sequential(nn.Linear(state_size, critic_hidden),
                       nn.ReLU(),
                       nn.Linear(critic_hidden, 1))

# method for efficient stochastic optimization, little mem req
#learning rate 
#Adam is taking in our critic parameters
critic_optimizer = Adam(critic.parameters(), lr=0.005)

#updates loss mean of our squared advantages diveded by 2
def update_critic(advantages):
    loss = .5 * (advantages ** 2).mean() #MSE
    critic_optimizer.zero_grad() #gradient
    loss.backward() #same as REINFORCE, this is back prop
    critic_optimizer.step()

#why are u here
    #magic number
# delta, maximum KL divergence
max_d_kl = 0.01

#feed in rollouts, which are tensors 
#and these are the trajectories
def update_agent(rollouts):
    #shoving the tensors together in dimension 0
    states = torch.cat([r.states for r in rollouts], dim=0)
    #getting flattened too
    actions = torch.cat([r.actions for r in rollouts], dim=0).flatten()

    #estimate our advantages of our rollouts/states
    advantages = [estimate_advantages(states, next_states[-1], rewards) for states, _, rewards, next_states in rollouts]
    advantages = torch.cat(advantages, dim=0).flatten()

    # Normalize advantages to reduce skewness and improve convergence
    advantages = (advantages - advantages.mean()) / advantages.std()  
    
    update_critic(advantages)

    #updated distribution of states we can take based on advantage
    distribution = actor(states)
    distribution = torch.distributions.utils.clamp_probs(distribution)
    probabilities = distribution[range(distribution.shape[0]), actions]

    # We will calculate the gradient wrt to the new probabilities (surrogate function),
    # so second probabilities should be treated as a constant
    #.detach() function detaches the computational graph
    #comp graphs are only generated for tensors that req gradients (requires.gradient = TRUE)
    #removing the comp graph means that it can be run in the CPU rather than the GPU
    L = surrogate_loss(probabilities, probabilities.detach(), advantages)
    KL = kl_div(distribution, distribution)

    parameters = list(actor.parameters())

    g = flat_grad(L, parameters, retain_graph=True)  # We will use the graph several times
    #making a new computation graph
    d_kl = flat_grad(KL, parameters, create_graph=True)  # Create graph, because we will call backward() on it (for HVP)

    #what is v
    #hierarchical view predictor?
    #Hessian vector product?
    #IDK
    def HVP(v):
        return flat_grad(d_kl @ v, parameters, retain_graph=True)

    search_dir = conjugate_gradient(HVP, g)
    max_length = torch.sqrt(2 * max_d_kl / (search_dir @ HVP(search_dir)))
    max_step = max_length * search_dir

    def criterion(step):
        # Apply parameters' update
        apply_update(step)

        #no_grad() takes away the gradient
        #making thing cheaper when you know what you will not have to back propogate anything
        with torch.no_grad():
            distribution_new = actor(states)
            distribution_new = torch.distributions.utils.clamp_probs(distribution_new)
            probabilities_new = distribution_new[range(distribution_new.shape[0]), actions]

            L_new = surrogate_loss(probabilities_new, probabilities, advantages)
            KL_new = kl_div(distribution, distribution_new)

        #checkin our improvment so our steps actually get better not worse
        L_improvement = L_new - L
        if L_improvement > 0 and KL_new <= max_d_kl:
            return True

        # Step size too big, reverse
        apply_update(-step)
        return False

    i = 0
    while not criterion((0.9 ** i) * max_step) and i < 10:
        i += 1






def estimate_advantages(states, last_state, rewards):
    values = critic(states)
    last_value = critic(last_state.unsqueeze(0))
    
    next_values = torch.zeros_like(rewards) #blank tensor the same size as rewards
    for i in reversed(range(rewards.shape[0])):
        last_value = next_values[i] = rewards[i] + 0.99 * last_value
        #broski what, IDK
        
    advantages = next_values - values
    return advantages

#this is the weird detach thing
#detaching means old probabilites? 
def surrogate_loss(new_probabilities, old_probabilities, advantages):
    #returns one value, the mean of improvement of the probabilities(new) times advantage
    return (new_probabilities / old_probabilities * advantages).mean()

#detach so it can't change during the calc
#computationally easier? 
def kl_div(p, q):
    p = p.detach()
    #sum(-1) possibly means takes the sum in the last dimension
    return (p * (p.log() - q.log())).sum(-1).mean()


def flat_grad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True

    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.view(-1) for t in g])
    return g

#this is an unconstrained iterative gradient solver 
#conjugate gradient method
#A is a passed in function, HVP in this case
#b should be a vector
#delta should be step size
#magic number
def conjugate_gradient(A, b, delta=0., max_iterations=10):
    x = torch.zeros_like(b)
    r = b.clone() #both be vectors
    p = b.clone()

    i = 0
    while i < max_iterations:
        AVP = A(p)

        dot_old = r @ r
        alpha = dot_old / (p @ AVP)

        x_new = x + alpha * p

        if (x - x_new).norm() <= delta:
            return x_new
        #this is something about if x_new or x is larger it seems 
        i += 1
        r = r - alpha * AVP

        beta = (r @ r) / dot_old
        p = r + beta * p

        x = x_new
    return x

#where is this going? it returns nothing and doesn't modify grad_flattened
# it modifies g which is not returned? IDK
def apply_update(grad_flattened):
    n = 0
    for p in actor.parameters():
        numel = p.numel()
        g = grad_flattened[n:n + numel].view(p.shape)
        p.data += g
        n += numel  #numel


# Train our agent
train(epochs=50, num_rollouts=10, render_frequency=50)