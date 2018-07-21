"""
Flappy Bird experiment using a feed-forward neural network.
"""
from __future__ import print_function

import os
import pickle
import numpy as np
import sys
import getopt

from flappy_bird import Game

import neat
import visualize
import pygame

runs_per_net = 5
visuals = False

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
    
        sim = Game(600,600,sys.argv[1])

        fitness = 0.0
        while True:
            inputs = sim.get_normalized_state()
            action = net.activate(inputs)

            # Apply action to the simulated bird
            valid = sim.step(np.argmax(action))

            # Stop if the network fails to keep the bird alive.
            # The per-run fitness is the traveled distance.
            if not valid:
                break
           
            fitness = sim.score

        fitnesses.append(fitness)

    # The genome's fitness is its worst performance across all runs.
    return min(fitnesses)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def main():

    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(4, eval_genome)
    winner = pop.run(pe.evaluate)

    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    visualize.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg")
    visualize.plot_species(stats, view=True, filename="feedforward-speciation.svg")

    node_names = {-1: 'x', -2: 'y', 0: 'wait', 1: 'flap'}

    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward.gv")

    
if __name__ == '__main__':
    main()