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
import torch.nn as nn

from torch.distributions import Categorical
from collections import namedtuple
from torch.optim import Adam

env = gym.make('CartPole-v0')

obs_size = env.observation_space.shape[0]
num_actions = env.action_space.n

#vine TRPO
#multiple rollouts to create traj
Rollout = namedtuple('Rollout',
                     ['states', 'actions', 'rewards', 'next_states', ])

#10 trajectory vine paths
def train(epochs=100, num_rollouts=10):
    for epoch in range(epochs):
        rollouts = []

        for t in range(num_rollouts):
            state = env.reset() #vine path we need reset the former vine path
            done = False #haven't done it yet

            samples = [] #collecting

            while not done:
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
        
        update_agent(rollouts)


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
critic = nn.Sequential(nn.Linear(obs_shape[0], critic_hidden),
                       nn.ReLU(),
                       nn.Linear(critic_hidden, 1))

# method for efficient stochastic optimization, little mem req
#learning rate 
#Adam is taking in our critic parameters
critic_optimizer = Adam(critic.parameters(), lr=0.005)

#updates loss mean of our squared advantages diveded by 2
def update_critic(advantages):
    loss = .5 * (advantages ** 2).mean() #weird function IDK
    critic_optimizer.zero_grad() #gradient
    loss.backward() #same as REINFORCE
    critic_optimizer.step()

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
    L = surrogate_loss(probabilities, probabilities.detach(), advantages)
    KL = kl_div(distribution, distribution)

    parameters = list(actor.parameters())

    g = flat_grad(L, parameters, retain_graph=True)  # We will use the graph several times
    d_kl = flat_grad(KL, parameters, create_graph=True)  # Create graph, because we will call backward() on it (for HVP)

    def HVP(v):
        return flat_grad(d_kl @ v, parameters, retain_graph=True)

    search_dir = conjugate_gradient(HVP, g, max_iterations=iterations)
    max_length = torch.sqrt(2 * delta / (search_dir @ HVP(search_dir)))
    max_step = max_length * search_dir

    def criterion(step):
        # Apply parameters' update
        apply_update(step)

        with torch.no_grad():
            distribution_new = actor(states)
            distribution_new = torch.distributions.utils.clamp_probs(distribution_new)
            probabilities_new = distribution_new[range(distribution_new.shape[0]), actions]

            L_new = surrogate_loss(probabilities_new, probabilities, advantages)
            KL_new = kl_div(distribution, distribution_new)

        L_improvement = L_new - L
        if L_improvement > 0 and KL_new <= delta:
            return True

        # Step size too big, reverse
        apply_update(-step)
        return False

    i = 0
    while not criterion((0.9 ** i) * max_step) and i < 10:
        i += 1
