import argparse
import numpy as np

import yaml

from src.models import NeuralNetwork, Fitness
from src.preprocessing import RayFrameParser

parser = argparse.ArgumentParser()
parser.add_argument("model_path", help="path to model, .npz")
args = parser.parse_args()

config = {}
with open('config.yaml', 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print(e)
        exit()
model = NeuralNetwork.load(args.model_path)
fitness = Fitness(np.random.randint(1000000), config["n_steps"] * 5, RayFrameParser())
print(fitness(model, display=True))
