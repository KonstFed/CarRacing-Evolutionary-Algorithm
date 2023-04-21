import gymnasium as gym
import cv2
import numpy as np
from random import random, randint
from tqdm import tqdm
import time
from multiprocessing import Pool
import matplotlib.pyplot as plt

class FrameParser():

    def __init__(self, mat='image',
                 l_u_corner=np.array([65, 45]),
                 r_d_corner=np.array([76, 50]),
                 ui_rotation_corners=np.array([[86, 37], [91, 58]]),
                 ui_speed_corners=np.array([[84, 13], [93, 13]])) -> None:
        # all coordinates are stored in format [y, x]
        self.l_u_corner = l_u_corner
        self.r_d_corner = r_d_corner
        self.mat = mat
        self.road_rgb = [105, 105, 105]

        self.s_p_forward1 = np.array(
            [l_u_corner[0], (l_u_corner[1] + r_d_corner[1])//2])
        self.s_p_forward2 = self.s_p_forward1 + [0, 1]

        self.s_p_left_side = np.array(
            [(l_u_corner[0] + r_d_corner[0]) // 2, l_u_corner[1]])
        self.s_p_rigt_side = np.array(
            [(l_u_corner[0] + r_d_corner[0]) // 2, r_d_corner[1]])

        self.s_p_left_angled = l_u_corner
        self.s_p_right_angled = np.array([l_u_corner[0], r_d_corner[1]])

        self.ui_rotation_corners = ui_rotation_corners
        self.ui_speed_coreners = ui_speed_corners

    def carCenter(self):
        center = (self.left_upper_corner + self.right_down_corner)//2
        return center

    def _ray(self, binary: np.ndarray, delta: np.array, start_pos: np.array, debug=False):
        if debug:
            out = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        count = 0
        pos = np.copy(start_pos)
        while (pos > [0, 0]).all() and (pos < binary.shape[:2]).all():
            if debug:
                out[pos[0], pos[1]] = (0, 0, 255)
                cv2.imshow("debug", out)
            if binary[pos[0], pos[1]] == 0:
                return count
            count += 1
            pos += delta

        return count

    def _getRays(self, binary):
        """
        Ray is distance to end of the road.
        Returns 5 rays: forward, left side, right side, left 45*, right 45*.
        """
        forward1 = self._ray(binary, np.array(
            [-1, 0]), self.s_p_forward1)
        forward2 = self._ray(binary, np.array([-1, 0]), self.s_p_forward2)
        forward = min(forward1, forward2)

        l_side = self._ray(binary, np.array([0, -1]), self.s_p_left_side)
        r_side = self._ray(binary, np.array([0, 1]), self.s_p_rigt_side)

        l_angled = self._ray(binary, np.array(
            [-1, -1]), self.s_p_left_angled) * np.sqrt(2)
        r_angled = self._ray(binary, np.array(
            [-1, 1]), self.s_p_right_angled) * np.sqrt(2)

        return [forward, l_side, r_side, l_angled, r_angled]

    def _binarizeWorld(self, frame):
        img = np.abs(frame - self.road_rgb)
        img = img.astype(np.uint8)
        out = np.zeros(img.shape[:2]).astype(np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i, j].sum() < 25:
                    out[i, j] = 255
                else:
                    out[i, j] = 0
        return out

    def _getRotation(self, frame):
        ui_rotation_frame = frame[self.ui_rotation_corners[0][0]: self.ui_rotation_corners[1]
                                  [0] + 1, self.ui_rotation_corners[0][1]:self.ui_rotation_corners[1][1] + 1]
        c_x = (self.ui_rotation_corners[0][1] + self.ui_rotation_corners[1]
               [1]) // 2 - self.ui_rotation_corners[0][1]

        binary = np.zeros(ui_rotation_frame.shape[:2])
        for y in range(ui_rotation_frame.shape[0]):
            for x in range(ui_rotation_frame.shape[1]):
                if ui_rotation_frame[y, x][1] > 40:
                    binary[y, x] = 255

        c_y = binary.shape[0] // 2
        count = 0
        for x in range(c_x+1, binary.shape[1]):
            if binary[c_y, x] == 0:
                break
            count += 1

        if count == 0:
            for x in range(c_x, -1, -1):
                if binary[c_y, x] == 0:
                    break
                count -= 1
        return count

    def _getSpeed(self, frame):
        ui_frame = frame[self.ui_speed_coreners[0][0]:self.ui_speed_coreners[1]
                         [0]+1, self.ui_speed_coreners[0][1]:self.ui_speed_coreners[1][1]+1]
        speed = 0
        for i in range(ui_frame.shape[0]-1, -1, -1):
            if ui_frame[i, 0][0] == 0:
                break
            speed += ui_frame[i, 0][0] / 255
        return speed

    def process(self, frame: np.array):
        """
        Process input image to input for GA.
        Returns array consisting of speed, rotation, forward ray, left side ray, right side ray, left 45 degree ray, right 45 degree ray.
        """
        world_frame = frame[:85,]  # remove bottom control panel
        binary = self._binarizeWorld(world_frame)
        rays = self._getRays(binary)
        rotation = self._getRotation(frame)
        speed = self._getSpeed(frame)
        return np.array([speed, rotation] + rays)

    def save(self, frame: np.array, filename='screen.png'):
        cv2.imwrite(filename, frame)


class NeuralNetwork():

    def __init__(self, n_input, n_output, hidden_sizes):
        sizes = [n_input] + hidden_sizes + [n_output]
        self.weight_mean = 0
        self.std = 2
        self.weights = []
        self.bias = []
        self.activations = []
        for i in range(1, len(sizes)):
            self.weights.append(np.random.normal(
                loc=self.weight_mean, scale=self.std, size=(sizes[i], sizes[i-1])))
            self.bias.append(np.random.normal(
                loc=self.weight_mean, scale=self.std, size=sizes[i]))
            self.activations.append(NeuralNetwork.relu)
        self.activations[-1] = NeuralNetwork.linear

    def create(weights, bias, activations):
        a = NeuralNetwork(0, 0, [1])
        a.weights = [np.copy(x) for x in weights]
        a.bias = [np.copy(x) for x in bias]
        a.activations = [x for x in activations]
        return a

    def copy(self):
        return NeuralNetwork.create(self.weights, self.bias, self.activations)

    def default():
        return NeuralNetwork(7, 5, [8, 6, 6])

    def relu(x):
        return x * (x > 0)

    def linear(x):
        return x

    def forward(self, x: np.array):
        out = np.copy(x)
        for i, weight in enumerate(self.weights):
            out = weight @ out
            out = out + self.bias[i]
            out = self.activations[i](out)
        return out

    def mutateLayer(self, layer_i, n):
        layer_shape = self.weights[layer_i].shape
        n_weights = self.weights[layer_i].size
        mutated = []
        i = 0
        while i < n:
            weight_i = randint(0, n_weights-1)
            if weight_i not in mutated:
                mutated.append(weight_i)
                new_weight = np.random.normal(
                    loc=self.weight_mean, scale=self.std, size=1)[0]
                new_bias = np.random.normal(
                    loc=self.weight_mean, scale=self.std, size=1)[0]

                self.weights[layer_i][weight_i//layer_shape[1],
                                      weight_i % layer_shape[1]] = new_weight
                self.bias[layer_i][weight_i//layer_shape[1]] = new_bias
            i += 1

    def mutate(self, layer_p: float, n_mutated):
        mutate_l_i = randint(0, len(self.weights)-1)
        self.mutateLayer(mutate_l_i, n_mutated)
        for layer_i in range(len(self.weights)):
            if layer_i != mutate_l_i and random() < layer_p:
                self.mutateLayer(layer_i, n_mutated)

    def cross(self, b):
        new = self.copy()
        layer_i = randint(0, len(new.weights)-1)
        new.weights[layer_i] = np.copy(b.weights[layer_i])
        new.bias[layer_i] = np.copy(b.bias[layer_i])
        new.mutate(0.3, 2)
        return new

    def load(path, n_layers=3):
        container = np.load(path)
        data = [container[x] for x in container]
        bias = data[n_layers+1:]
        weights = data[:n_layers+1]
        activations = [NeuralNetwork.relu for x in range(len(weights))]
        activations[-1] = NeuralNetwork.linear
        return NeuralNetwork.create(weights, bias, activations)

    def save(self, path):
        data = self.weights + [x for x in self.bias]
        np.savez(path, *data)


class GeneticAlgorithm():

    def __init__(self, population_size, fitness, n_processes) -> None:
        self.population_size = population_size
        self.population = [NeuralNetwork.default()
                           for x in range(population_size)]
        self.fitness = fitness
        self.n_processes = n_processes

    def evolutionStep(self):
        new = []
        for i, model in enumerate(self.population):
            cur = model.copy()
            cur.mutate(0.3, 2)
            new.append(cur)

        for i in range(0, len(self.population), 2):
            new.append(self.population[i].cross(self.population[i+1]))
            new.append(self.population[i+1].cross(self.population[i]))

        self.population = self.population + new
        p = Pool(self.n_processes)
        fitness_value = p.map(fitness, self.population)
        fitness_value = [(self.population[x], fitness_value[x])
                         for x in range(len(self.population))]
        fitness_value.sort(key=lambda x: x[1], reverse=True)
        self.population = list(
            map(lambda x: x[0], fitness_value[:self.population_size]))

    def evolve(self, n_iterations=100, verbose=True):
        if verbose:
            iter = tqdm(range(n_iterations))
            step_execution_time = []
            best_fitnesses = []
        else:
            iter = range(n_iterations)

        for i in iter:
            if verbose:
                st = time.time()
            self.evolutionStep()
            if verbose:
                best_fitnesses.append(self.fitness(self.population[0]))
                ft = time.time()
                step_execution_time.append(ft - st)

        if verbose:
            print(f"Average execution time:{np.mean(step_execution_time)} s.")
            for i in range(len(step_execution_time)):
                print(f"step {i}: {step_execution_time[i]}s.")
            print("Fitness progression")
            print(", ".join(best_fitnesses))
        return self.population[0]


def fitness(model: NeuralNetwork, n_steps=300, display=False):
    fp = FrameParser()
    if display:
        env = gym.make("CarRacing-v2", continuous=False, render_mode="human",
                       domain_randomize=False)
    else:
        env = gym.make("CarRacing-v2", continuous=False,
                       domain_randomize=False)
    total_reward = 0
    observation, info = env.reset()
    for i in range(n_steps):
        parsed_input = fp.process(observation)
        on_grass = 1 if sum(parsed_input[2:]) == 0 else 0
        total_reward += parsed_input[0] - 1 - on_grass * 10
        # 0 - nothing, 1 - steer right, 2 - steer left, 3 - gas, 4 - brake
        action = np.argmax(model.forward(parsed_input))
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.close()
            return total_reward - 10000
    env.close()
    return total_reward
