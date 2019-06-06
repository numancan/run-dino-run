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

generation=1
def eval_genomes(genomes, config):
    global generation

    for genome_id, genome in genomes:

        show_nn(config, genome)
        old_fitness = genome.fitness

        if len(genome.connections.values()) == 0:
            genome.fitness = 0
        else:
            result = play(genome, config)
            genome.fitness = result

            # If genome live more than 2 minute, save with id and life time
            # Example best_id_lifeTime
            if result > 120:
                pickle.dump(genome, open('best-genomes/best_{0}_{1}.pkl'.format(genome_id, result), 'wb'))

        print(  
                "\n"*10,
                "*"*50,
                "\n Generation: {0}\n".format(generation),
                "\n Player ID: {0} Old Fitness: {1} New Fitness: {2}\n".format(genome_id, old_fitness, genome.fitness),
                "*"*50+"\n",
                "\n"*10
             )

    generation+=1


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    print("Train Starting in 3 sec !")
    time.sleep(3)

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 50)

    pickle.dump(winner, open('best-genomes/winner.pkl', 'wb'))

    # Visualize winner
    node_names = {-1: 'Distance', -2: 'Gap', -3: 'Speed', 0: 'Duck', 1: 'Jump'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
