import numpy as np
import copy
import torch
import sys
import time
from os.path import join, exists
from os import mkdir

from fitness_functions import *


def compute_ranks(x):
  """
  Returns rank as a vector of len(x) with integers from 0 to len(x)
  """
  assert x.ndim == 1
  ranks = np.empty(len(x), dtype=int)
  ranks[x.argsort()] = np.arange(len(x))
  return ranks

def compute_centered_ranks(x):
  """
  Maps x to [-0.5, 0.5] and returns the rank
  """
  y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
  y /= (x.size - 1)
  y -= .5
  return y


class EvolutionStrategyStatic(object):
    def __init__(self, weights, generator, generator_init_params, restricted, sigma=0.1, learning_rate=0.2, decay=0.995):
        
        self.weights = weights
        self.POPULATION_SIZE = generator_init_params['population_size']
        self.SIGMA = sigma
        self.learning_rate = learning_rate
        self.decay = decay
        self.num_threads = 1
        self.update_factor = self.learning_rate / (self.POPULATION_SIZE * self.SIGMA)
        self.generator = generator
        self.generator_init_params = generator_init_params
        self.restricted = restricted
        self.choice_batch=generator_init_params['choice_batch']
        
        if generator == 'MLP':
            if self.choice_batch>1:
                self.get_reward = fitness_MLP_alltogether #fitness_MLP_batch
            else:
                self.get_reward = fitness_MLP
        elif generator == 'SymMLP':
            if self.choice_batch>1:
                self.get_reward = fitness_SymMLP_alltogether #fitness_MLP_batch
            else:
                self.get_reward = fitness_SymMLP
        elif generator == 'RNN':
            if self.choice_batch>1:
                self.get_reward = fitness_RNN_alltogether
            else:
                print("Single structure Rating")
                self.get_reward = fitness_RNN
        
    def _get_weights_try(self, w, p):
        
        if self.SIGMA != 0:
            weights_try = []
            for index, i in enumerate(p):
                jittered = self.SIGMA * i
                weights_try.append(w[index] + jittered)
            weights_try = np.array(weights_try)
        elif self.SIGMA == 0:
            weights_try = np.array(p)

        return weights_try   # weights_try[i] = w[i] + sigma * p[i]
 
    def get_weights(self):
        return self.weights

    def _get_population(self):
        population = []
        for i in range(int(self.POPULATION_SIZE/2) ):
            x = []
            x2 = []
            for w in self.weights:
                j = np.random.randn(*w.shape)
                x.append(j)
                x2.append(-j) 

            population.append(x)
            population.append(x2)
            
        population = np.array(population)

        return population  


    def _get_rewards(self, population):
        
        batch_size = self.generator_init_params['choice_batch']
        assert batch_size > 0 #minimum 1 ...

        # Single-core
        rewards = []
        if batch_size>1:
            all_weights = np.stack([np.array(self._get_weights_try(self.weights, p)) for p in population], axis=0) 
            rewards=self.get_reward(all_weights, self.generator_init_params, self.restricted)

        else:
            for p in population: 
                weights_try = np.array(self._get_weights_try(self.weights, p))   # weights_try[i] = self.weights[i] + sigma * p[i]
                reward=self.get_reward(weights_try, self.generator_init_params, self.restricted)
                rewards.append(reward)
        
        rewards = np.array(rewards)
        emptyness = True if np.sum(rewards) == 0 else False

        return rewards, emptyness

    def _update_weights(self, rewards, population): 
        
        rewards = compute_centered_ranks(rewards)   # Project rewards to [-0.5, 0.5]

        std = rewards.std()
        if std == 0:
            raise ValueError('Variance should not be zero')


        rewards = (rewards - rewards.mean()) / std  # Normalize rewards
        for index, w in enumerate(self.weights):
            layer_population = np.array([p[index] for p in population])   # Array of all weights[i] for all the networks in the population

            self.update_factor = self.learning_rate / (self.POPULATION_SIZE * self.SIGMA)
            self.weights[index] = w + self.update_factor * np.dot(layer_population.T, rewards).T 

        if self.update_factor > 0.001: 
            self.learning_rate *= self.decay

        #Decay sigma
        if self.SIGMA>0.001:
            self.SIGMA *= 0.999


    def run(self, generations, path='weights'):
        
        id_ = str(int(time.time()))
        if not exists(path + '/' + id_):
            mkdir(path + '/' + id_)
            
        print('\n********************\n \nRUN: ' + id_ + '\n\n********************\n')
            
        
        generation = 0
        while generation < generations:                                    # Algorithm 2. Salimans, 2017: https://arxiv.org/abs/1703.03864
            
            population = self._get_population()                            # List of list of random nets [[w1, w2, .., w122888],[...],[...]] : Step 5
            rewards, emptyness = self._get_rewards(population)             # List of corresponding rewards for self.weights + jittered populations : Step 6
            

            if emptyness and generation == 0:
                self._update_weights(rewards, population)                     

            if not emptyness:
                self._update_weights(rewards, population)                     
                print('iter %4i | update_factor: %f  lr: %f  sigma: %f | sum_w: %i sum_abs_w: %i' % ( generation + 1, self.update_factor, self.learning_rate, self.SIGMA, int(np.sum(self.weights)) ,int(np.sum(abs(self.weights))) ), flush=True)
                torch.save(self.get_weights(), path + "/"+ id_ + "/" + str(self.generator) + "__restricted_" + str(self.restricted) + "__sigma_" + str(self.SIGMA)[:4] + "__lr_" + str(self.learning_rate)[:6] + "__gen_{}.dat".format(generation))  
                
                generation += 1
            
            