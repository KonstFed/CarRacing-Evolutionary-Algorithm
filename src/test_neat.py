import pickle
import os
from random import randint

import neat

from models import NeatModel, Fitness
from preprocessing import BinaryFrameParser, RayFrameParser

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'neat_config')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)

with open("best_models/neat/winner.pkl", "rb") as f:
    genome = pickle.load(f)

model = NeatModel(-1, genome, config)
fitness = Fitness(randint(0, 1_000_000), 1000, RayFrameParser())
fitness(model, display=True)