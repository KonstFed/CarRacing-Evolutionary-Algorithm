import pickle
import os
import sys
from random import randint

import neat

from src.models import NeatModel, Fitness
from src.preprocessing import BinaryFrameParser, RayFrameParser


if sys.argv[2] == "binary":
    parser = BinaryFrameParser()
else:
    parser = RayFrameParser()

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'configs/neat_config_' + sys.argv[2])
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)



with open(sys.argv[1], "rb") as f:
    genome = pickle.load(f)

model = NeatModel(-1, genome, config)
fitness = Fitness(randint(0, 1_000_000), 2000, parser)
print(fitness(model, display=True))