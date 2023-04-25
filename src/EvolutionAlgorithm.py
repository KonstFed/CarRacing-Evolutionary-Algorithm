import time

import gymnasium as gym
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

from src.models import NeuralNetwork
from src.preprocessing import FrameParser

class GeneticAlgorithm():

    def __init__(self, population_size, fitness, n_processes, n_steps) -> None:
        self.population_size = population_size
        self.population = [NeuralNetwork.default()
                           for x in range(population_size)]
        self.fitness = fitness
        self.n_processes = n_processes
        GeneticAlgorithm.n_steps = n_steps
        GeneticAlgorithm.env_seed = np.random.randint(1000000)


    def evolutionStep(self):
        new = []
        for model in self.population:
            cur = model.copy()
            cur.mutate(0.3, 2)
            new.append(cur)

        for i in range(0, len(self.population), 2):
            new.append(self.population[i].cross(self.population[i+1]))
            new.append(self.population[i+1].cross(self.population[i]))

        self.population = self.population + new
        p = Pool(self.n_processes)
        GeneticAlgorithm.env_seed = np.random.randint(1000000)
        fitness_value = p.map(self.fitness, self.population)
        fitness_value = [(self.population[x], fitness_value[x])
                         for x in range(len(self.population))]
        fitness_value.sort(key=lambda x: x[1], reverse=True)
        self.population = list(
            map(lambda x: x[0], fitness_value[:self.population_size]))

    def evolve(self, n_iterations=100, verbose=True):
        if verbose:
            iter = tqdm(range(n_iterations))
            step_execution_time = []
        else:
            iter = range(n_iterations)

        for i in iter:
            if verbose:
                st = time.time()
            self.evolutionStep()
            if verbose:
                ft = time.time()
                step_execution_time.append(ft - st)

        if verbose:
            print(f"Average execution time:{np.mean(step_execution_time)} s.")
            for i in range(len(step_execution_time)):
                print(f"step {i}: {step_execution_time[i]}s.")
        return self.population[0]


def fitness(model, display=False):
    if display:
        env = gym.make("CarRacing-v2", domain_randomize=False, render_mode="human")
    else:
        env = gym.make("CarRacing-v2", domain_randomize=False)

    fp = FrameParser()

    total_reward = 0
    observation, info = env.reset(seed=GeneticAlgorithm.env_seed)
    for i in range(GeneticAlgorithm.n_steps):
        parsed_input = fp.process(observation)
        on_grass = 1 if sum(parsed_input[2:]) == 0 else 0
        total_reward += parsed_input[0] - on_grass * 10
        action = model.forward(parsed_input)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            if reward < 0:
                total_reward -= 1000
            break

    env.close()

    return total_reward
