import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from collections import deque
import time
import random

from gymnasium import spaces

class OneMoleculeEnv(gym.Env):
    def __init__(self, initial_value=10, molecule_lifetime = 1, dt = 0.1, max_steps=100, history_length=5, target_value=10, obs_cap=100, render_mode=None):
        super(OneMoleculeEnv, self).__init__()

        self.initial_value = initial_value
        self.prob_death = 1 - np.exp(-dt/molecule_lifetime)
        self.dt = dt
        self.max_steps = max_steps
        self.history_length = history_length
        self.target_value = target_value
        self.obs_cap = obs_cap  # Cap for the observation space
        self.render_mode = render_mode

        self.current_value = initial_value
        self.current_step = 0
        self.decays = 0
        self.ideal_action = 0

        # Define action and observation space
        self.action_space = spaces.Discrete(obs_cap)  # the number of molecules to send in ranges from 0 to the cap
        self.observation_space = spaces.Box(
            low=0, high=obs_cap, shape=(history_length,), dtype=np.float32
        )

        # Initialize the history of values
        self.history = np.full(history_length, initial_value, dtype=np.float32)

        # Pre-generate random numbers to avoid generating them at each step
        self.random_numbers = np.random.rand(max_steps)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            self.random_numbers = np.random.rand(self.max_steps)  # Re-generate random numbers if seeded
        self.current_value = self.initial_value
        self.current_step = 0
        self.history = np.full(self.history_length, self.initial_value, dtype=np.float32)
        return self.history, {}

    def _ensure_random_numbers(self):
        # Reset the random numbers if current_step exceeds max_steps
        if self.current_step >= self.max_steps:
            self.random_numbers = np.random.rand(self.max_steps)
            self.current_step = 0  # Reset current step to start fresh

    # First, ensure enough random numbers are available
    # The probability a molecule has decayed in the time interval given is dependent on dt and the lifetime.
    # Decrease the value with a number drawn from a binomial distribution
    # This reflects each molecule having a finite probability of decaying in the timestep
    # The probability a molecule has decayed in the time interval given is dependent on dt and the lifetime.
    # Then add molecules.
    # We then update the history, increase the steps, and calculate the error (reward)
    # Then return the standard gymnasium properties.
    def step(self, action):
        # Ensure enough random numbers are available
        self._ensure_random_numbers()

        self.decays = -np.random.binomial((int)(self.current_value), self.prob_death, 1).item()
        self.current_value += self.decays
        self.ideal_action = max(self.target_value - self.current_value, 0)
        self.current_value += action
        #print ("The options for self current value are ")
        #print (self.current_value, ' ', self.obs_cap)
        #print (self.obs_cap, ' vs ', self.current_value)

        self.current_value = min(self.current_value, self.obs_cap)

        # Shift elements towards the end
        self.history[1:] = self.history[:-1]
        
        # Insert the most recent value at the 0th index
        self.history[0] = np.float32(self.current_value)  # Ensure it's a scalar

        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        reward = -float((self.current_value - self.target_value) ** 2)  # Ensure the reward is a float

        return self.history, reward, done, done, self.ideal_action

    def render(self):
        if self.render_mode == 'human':
            print(f"Step: {self.current_step}, Value: {self.current_value}, History: {self.history}")

    def close(self):
        pass

class ReplayMemory: ##Replay memory to store past experiences to learn from
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def add(self, experience):
        self.memory.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class NeuralNetwork(nn.Module):
    def __init__(self, layers_array):
        """
        Initialize the neural network with a customizable architecture.

        Parameters:
        layers_array (list of int): Array where each element specifies the number of neurons in each layer.
                                    The first element is the input dimension, and the last element is the output dimension.
        """
        super(NeuralNetwork, self).__init__()

        # Ensure the array has at least two elements (input layer and one output layer)
        if len(layers_array) < 2:
            raise ValueError("The layers_array must contain at least two elements: input and output layers.")

        # Create a list to hold the layers
        layers = []

        # Iterate through the layers_array to create the network layers
        for i in range(len(layers_array) - 1):
            input_size = layers_array[i]
            output_size = layers_array[i + 1]
            
            # Add a linear layer
            layers.append(nn.Linear(input_size, output_size))
            
            # Add a ReLU activation function after each linear layer, except the last one
            if i < len(layers_array) - 2:
                layers.append(nn.ReLU())

        # Combine all layers into a Sequential module
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear_relu_stack(x)


def step_function(model, optimizer, loss_fn, states, outs):
    model.train()
    
    predict_output = model(states)
    loss = loss_fn(predict_output, outs)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

##Our memory contains states and the optimal output value
def train_from_memory(model, optimizer, loss_fn, replay_memory, batch_size):
    if len(replay_memory) < batch_size:
        return 0

    batch = replay_memory.sample(batch_size)
    state_list, output_list = zip(*batch)

    states = torch.tensor(np.array(state_list), dtype=torch.float32)
    outputs = torch.tensor(np.array(output_list), dtype=torch.float32).unsqueeze(1)

    loss = step_function(model, optimizer, loss_fn, states, outputs)
    return loss

def train_molecule_controller(model, optimizer, loss_fn, replay_memory, steps, target, molecule_lifetime, dt, history_length, observable_indices, RUN_SEED, batch_size):
    ##Here we choose our environment
    env = OneMoleculeEnv(
        initial_value=target,
        molecule_lifetime=molecule_lifetime,
        dt=dt,
        max_steps=steps,
        history_length=history_length,
        target_value=target,
        obs_cap=target*3,
        render_mode=None
    )
    
    RUN_SEED = 0
    
    observation, info = env.reset(seed=RUN_SEED)
    observation_tensor = torch.from_numpy(observation[observable_indices])

    for step in range(steps):

        ##we compute our optimal action using the most 
        #print (observation[observable_indices])
        with torch.no_grad():
            action = model(observation_tensor)
        action[action <0] = 0 ##controller cannot remove molecules
        rounded_action = torch.round(action).int().item() ##round the action to the nearest integer
        observation, reward, done, truncated, info = env.step(rounded_action)
        observation_tensor =  torch.from_numpy(observation[observable_indices])
        #print ("Action taken ", rounded_action, ' ideal action ', info, ' current observation ', observation_tensor)
        replay_memory.add((observation_tensor, info))

        q_error = train_from_memory(model, optimizer, loss_fn, replay_memory, batch_size)
    env.close()
    return 0

def test_molecule_controller(model, optimizer, loss_fn, replay_memory, steps, target, molecule_lifetime, dt, history_length, observable_indices, RUN_SEED, batch_size):
    ##Here we choose our environment
    env = OneMoleculeEnv(
        initial_value=target,
        molecule_lifetime=molecule_lifetime,
        dt=dt,
        max_steps=steps,
        history_length=history_length,
        target_value=target,
        obs_cap=target*3,
        render_mode=None
    )
    
    rewards = []
    training_error = []
    RUN_SEED = 0
    
    observation, info = env.reset(seed=RUN_SEED)
    observation_tensor =  torch.from_numpy(observation[observable_indices])
    
    for step in range(steps):

        ##we compute our optimal action using the most 
        #print (observation[observable_indices])
        with torch.no_grad():
            action = model(observation_tensor) #unnormalize our actions
        action[action <0] = 0 ##controller cannot remove molecules
        rounded_action = torch.round(action).int().item() ##round the action to the nearest integer
        observation, reward, done, truncated, info = env.step(rounded_action)
        observation_tensor =  torch.from_numpy(observation[observable_indices])
        #print ("Action taken ", rounded_action, ' ideal action ', info, ' current observation ', observation_tensor)
        replay_memory.add((observation_tensor, info))

        q_error = train_from_memory(model, optimizer, loss_fn, replay_memory, batch_size)
        if q_error > 0:
            training_error.append(q_error)
        rewards.append(reward)
    env.close()
    return np.array(rewards), np.array(training_error)

def optimal_action_th(molecules, target, molecule_lifetime, dt):
    prob_survival = np.exp(-dt/molecule_lifetime)
    optimal_action = target - molecules*prob_survival
    return optimal_action

def optimal_solution(steps, target, molecule_lifetime, dt = 0.5, RUN_SEED = 0):
    ##Here we choose our environment
    env = OneMoleculeEnv(
        initial_value=target,
        molecule_lifetime=molecule_lifetime,
        dt=dt,
        max_steps=steps,
        history_length=1,
        target_value=target,
        obs_cap=target*3,
        render_mode=None
    )
        
    rewards = []
    observation, info = env.reset(seed=RUN_SEED)
    
    for step in range(steps):

        ##we compute our optimal action using the most 
        action = optimal_action_th(observation[-1], target, molecule_lifetime, dt)
        rounded_action = (int)(np.round(action))
        observation, reward, done, truncated, info = env.step(rounded_action)
        #print ("Reward ", reward, " Observation ", observation, " Action ", action)

        ##The error is the negative of the reward
        rewards.append(reward)
    env.close()
    return np.array(rewards)


def control_plot_1d(model, upper_mol, molecule_lifetime, dt):
    action_vals = np.zeros(upper_mol)
    for cyc in range(upper_mol):
        mol = torch.tensor([cyc], dtype=torch.float32)
        action_vals[cyc] = model(mol)
    return action_vals

def control_plot_1dopt(upper_mol, target, molecule_lifetime, dt):
    action_vals = np.zeros(upper_mol)
    for cyc in range(upper_mol):
        action_vals[cyc] = max(optimal_action_th(cyc, target, molecule_lifetime, dt), 0)
    return action_vals


# Initialization
def table_init(dimensions, size, initial_value):
    action_array = np.full((size,) * dimensions, initial_value, dtype=np.float32)  # Array to store parameters
    counter_array = np.zeros((size,) * dimensions, dtype=np.float32)  # Array to store values to be updated
    return action_array, counter_array

def learning_rate(counter, forget_param):
    #max(np.exp(-counter*0.1), 0.2)
    return max(1.0/(counter + 1), forget_param)

# Function to update array2 based on a parameter from array1
def table_learn(action_array, counter_array, forget_param, index, new_value):
    # Get the parameter from array1
    counter = counter_array[index]

    ##we can use this counter to more intelligently update our array
    ##but lets not right now!
    lr = learning_rate(counter, forget_param)

    # Update array2 based on the parameter
    # Example: Add the parameter to the value and store it in array2
    action_array[index] = action_array[index]*(1 - lr) + lr*new_value
    counter_array[index] += 1
    #print (lr)

def tabular_molecule_controller(action_array, counter_array, forget_param, steps, target, molecule_lifetime, dt, history_length, observable_indices, RUN_SEED):
    ##Here we choose our environment
    env = OneMoleculeEnv(
        initial_value=target,
        molecule_lifetime=molecule_lifetime,
        dt=dt,
        max_steps=steps,
        history_length=history_length,
        target_value=target,
        obs_cap=target*3,
        render_mode=None
    )
    rewards = []
    observation, info = env.reset(seed=RUN_SEED)
    for step in range(steps):
        action_index = tuple(observation[observable_indices].astype(int)) # tuple allows us to use this for numpy array slicing
        action = action_array[action_index]
        rounded_action = np.round(action)

        observation, reward, done, truncated, info = env.step(rounded_action)
        table_learn(action_array, counter_array, forget_param, action_index, info)
        rewards.append(reward)
    env.close()
    return np.array(rewards)

