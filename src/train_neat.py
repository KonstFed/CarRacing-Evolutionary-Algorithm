"""
2-input XOR example -- this is most likely the simplest possible example.
"""

import os
import pickle
from multiprocessing import Pool
from random import randint

import neat

from models import Fitness, NeatModel
from preprocessing import BinaryFrameParser, RayFrameParser


n_steps = 700


def eval_genomes(genomes, config):
    fitness = Fitness(randint(0, 1_000_000), n_steps, RayFrameParser())
    models = [NeatModel(id, genome, config) for id, genome in genomes]
    p = Pool(10)
    fitness_value = p.map(fitness, models)
    for i in range(len(models)):
        models[i].genome.fitness = fitness_value[i]


def save(genome):
    with open("best_models/neat/winner.pkl", "wb") as f:
        pickle.dump(genome, f)
        f.close()


def run(config_file):
    # Load configuration.
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    # stats = neat.StatisticsReporter()
    # p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5, filename_prefix="neat_logs/checkpoints/neat-checkpoint-"))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 20)
    save(winner)


if __name__ == "__main__":
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat_config")
    run(config_path)
