# NEAT-Flappy-Bird

NeuroEvolution of Augmenting Topologies(NEAT) is a technique used in machine learning, which mimicks the process of natural selection to evolve a population of AI candidates which can solve a given task.

Read more about NEAT here:  
http://eplex.cs.ucf.edu/hyperNEATpage/

For this experiment I use NEAT-Python for the NEAT implementation and Pygame for the game logic:  
https://neat-python.readthedocs.io/en/latest/  
https://www.pygame.org/docs/

I wrote my own Flappy Bird clone and use the NEAT implementation to train my population.  
I use the x distance to the next pipe and the y distance to the next hole as inputs for the neural network.  
We have two possible output actions: Wait or Flap.   

First run:   
flappy_bird_neat.py <visualize=True/False>

Then:  
flappy_bird.py

Be warned that enabling visualized training will slow down the process a lot.