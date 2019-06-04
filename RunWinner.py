import os
import neat
import pickle
import time
from Train import ShowNN
from DinoRun import Play


def Run(config):
    genome = pickle.load(open('winners/winner_222_13431.pkl', 'rb'))
    ShowNN(config, genome)
    print("Starting 3 sec!")
    time.sleep(3)
    Play(genome, config)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    Run(config)
