import argparse

import yaml

from CarRacing import fitness
from models import NeuralNetwork

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
fitness(model, display=True, n_steps=2000)
