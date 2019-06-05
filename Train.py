import cv2
import numpy as np
import pickle
import time
import os
import neat
import visualize
from rundino import play


def show_nn(config, genome):
    node_names = {-1: 'Distance', -2: 'Gap', -3: 'Speed', 0: 'Duck', 1: 'Jump'}
    visualize.draw_net(config, genome, True, node_names=node_names, filename="graph/nn")
    cv2.namedWindow("NN")
    cv2.moveWindow("NN", 50, 350)
    cv2.imshow("NN", cv2.imread("graph/nn.png"))


def eval_genomes(genomes, config):

    for genome_id, genome in genomes:

        show_nn(config, genome)
        old_fitness = genome.fitness

        if len(genome.connections.values()) == 0:
            genome.fitness = 0
        else:
            result = play(genome, config)
            genome.fitness = result
            if result > 120:
                pickle.dump(genome, open('best-genomes/best_{0}_{1}.pkl'.format(genome_id, result), 'wb'))

        print("Player ID: {0} Old Fitness: {1} New Fitness: {2}".format(genome_id, old_fitness, genome.fitness))


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)

    print("Train Starting in 3 sec !")
    time.sleep(3)
    # Train for 30 generation
    winner = p.run(eval_genomes, 30)

    pickle.dump(winner, open('winner.pkl', 'wb'))
    show_nn(config, winner)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
