import os
import sys
import pickle
from multiprocessing import Pool
from random import randint

import neat

from src.models import Fitness, NeatModel
from src.preprocessing import BinaryFrameParser, RayFrameParser
import visualize

n_steps = 700

parser = None


def eval_genomes(genomes, config):
    global parser
    fitness = Fitness(randint(0, 1_000_000), n_steps, parser)
    models = [NeatModel(id, genome, config) for id, genome in genomes]
    p = Pool(10)
    fitness_value = p.map(fitness, models)
    for i in range(len(models)):
        models[i].genome.fitness = fitness_value[i]


def save(genome, input_mode: str):
    with open("best_models/neat/" + input_mode + "_current.pkl", "wb") as f:
        pickle.dump(genome, f)
        f.close()


def run(config_file, input_mode: str):
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
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(
        neat.Checkpointer(5, filename_prefix="neat_logs/checkpoints/neat-checkpoint-")
    )
    # node_names = {-1}
    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 20)
    save(winner, input_mode)
    node_names = {
        -1: "speed",
        -2: "angle",
        -3: "rotation",
        -4: "gr_forward",
        -5: "gr_left_side",
        -6: "gr_right_side",
        -7: "gr_left_45",
        -8: "gr_right_45",
        -9: "rd_forward",
        -10: "rd_left_side",
        -11: "rd_right_side",
        -12: "rd_left_45",
        -13: "rd_right_45",
        0: "steer",
        1: "gas/break",
    }
    visualize.draw_net(
        config,
        winner,
        False,
        node_names=node_names,
        filename="neat_logs/neat_stats/net.gv",
    )


if __name__ == "__main__":
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    parser_config = sys.argv[1]
    if parser_config == "ray":
        parser = RayFrameParser()
    elif parser_config == "binary":
        parser = BinaryFrameParser()

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "configs/neat_config_" + sys.argv[1])
    run(config_path, parser_config)
