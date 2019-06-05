import os
import neat
import pickle
import time
from train import show_nn
from rundino import play


def run(config):
    genome = pickle.load(open('winner.pkl', 'rb'))
    show_nn(config, genome)
    print("Starting 3 sec!")
    time.sleep(3)
    play(genome, config)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)


    run(config)
