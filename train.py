import argparse

import yaml

from src.EvolutionAlgorithm import GeneticAlgorithm, Fitness


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file", help="Where to save best model, .npz file")
    args = parser.parse_args()
    config = {}
    with open('config.yaml', 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)
            exit()

    population_size = config["population_size"]
    n_processes = config["n_processes"]
    n_steps = config["n_steps"]
    ga = GeneticAlgorithm(population_size, Fitness, n_processes, n_steps)
    print(f"Number of processes: {n_processes}")
    best = ga.evolve(n_iterations=config["number_iterations"])
    best.save(args.output_file)
