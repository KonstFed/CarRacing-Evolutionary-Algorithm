import time

import gymnasium as gym
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

from src.models import NeuralNetwork
from src.preprocessing import RayFrameParser

class GeneticAlgorithm():

    def __init__(self, population_size, fitness_class, n_processes, n_steps) -> None:
        self.population_size = population_size
        self.population = [NeuralNetwork.default()
                           for x in range(population_size)]
        self.fitness = fitness_class(126, n_steps)
        self.n_processes = n_processes

    def evolutionStep(self):
        new = []
        for model in self.population:
            cur = model.copy()
            cur.mutate(0.3, 2)
            new.append(cur)

        for i in range(0, len(self.population), 2):
            offspring1 = self.population[i].cross(self.population[i+1])
            offspring2 = self.population[i+1].cross(self.population[i])
            offspring1.mutate(0.3, 2)
            offspring2.mutate(0.3, 2)
            new.append(offspring1)
            new.append(offspring2)

        self.population = self.population + new
        self.fitness.env_seed = np.random.randint(1000000)
        p = Pool(self.n_processes)
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
            step_best_finess = []
        else:
            iter = range(n_iterations)

        for i in iter:
            if verbose:
                st = time.time()
            self.evolutionStep()
            if verbose:
                ft = time.time()
                step_execution_time.append(ft - st)
                step_best_finess.append(self.fitness(self.population[0]))

        if verbose:
            print(f"Average execution time:{np.mean(step_execution_time)} s.")
            for i in range(len(step_execution_time)):
                print(f"step {i}: {step_execution_time[i]}s.")
                print(f"fitness: {step_best_finess[i]}.")
        return self.population[0]


class Fitness:
    def __init__(self, env_seed, n_steps, period=3, rot_reduction=0.5):
        self.env_seed = env_seed
        self.n_steps = n_steps
        self.period = period
        self.rot_reduction = rot_reduction 

    def __call__(self, model, display=False):
        if display:
            env = gym.make("CarRacing-v2", domain_randomize=False, render_mode="human", max_episode_steps=self.n_steps)
        else:
            env = gym.make("CarRacing-v2", domain_randomize=False, max_episode_steps=self.n_steps)

        fp = RayFrameParser()

        total_reward = 0
        count = 0
        action = None
        observation, info = env.reset(seed=self.env_seed)
        for i in range(self.n_steps):
            if count % self.period == 0:
                parsed_input = fp.process(observation)
                output = model.forward(parsed_input)
                action = [output[0], output[1] if output[1] > 0 else 0, -output[1] if output[1] < 0 else 0]
                # action = [output[0] * 2 - 1, output[1], output[2]]

                if sum(parsed_input[3:9]) == 0:
                    total_reward -= 0.3

                # print(parsed_input)
            else:
                action = [action[0] * self.rot_reduction, 0, 0]
            
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_reward -= abs(action[0]) * action[1]

            count += 1

            if terminated or truncated:
                break

        env.close()

        return total_reward
