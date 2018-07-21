import pygame
import time
import random
import os
import neat
import pickle
import numpy as np
import PAdLib as padlib

class Bird:
    
    def __init__(self, x, y):
        self.y = y
        self.x = x
        self.velocity = 0
        self.diameter = 40
        self.cooldown = 0
        
        
class Pipe:
    
    def __init__(self, x, y, gap_size):
        self.gap_size = gap_size
        self.y = y
        self.x = x
        self.width = 60

        
class Game:

    def __init__(self, width, height, visualize=False, Player=None):
        self.width = width
        self.height = height
        self.gravity = .5
        self.strength = -8
        self.visualize = visualize
        self.score = 0
        self.pipes = []
        self.pipe_cooldown = 0
        self.gap_size = 160
        self.bird = None
        if visualize:
            pygame.init()
            self.screen = pygame.display.set_mode((width+400,height))
            self.screen.fill(pygame.Color('black'))
            pygame.display.set_caption('Flappy Bird')
        self.reset()
        
    def reset(self):
        self.score = 0
        self.bird = Bird(self.width//3, self.height//2)
        
    def flap(self):
        self.bird.velocity = self.strength
        self.bird.cooldown = 5
        
    def step(self,action):
        if self.visualize:
            pygame.event.pump()
            time.sleep(0.05)
        if self.pipe_cooldown <= 0:
            self.pipes.append(Pipe(self.width, random.randint(0, self.height-self.gap_size), self.gap_size))
            self.pipe_cooldown = 160
        else:
            self.pipe_cooldown -= 1
        for p in self.pipes:
            p.x -= 2
        self.pipes = [p for p in self.pipes if p.x >= -p.width]
        if action == 1 and self.bird.cooldown == 0:
            self.flap()
        elif self.bird.cooldown > 0:
            self.bird.cooldown -= 1

        self.bird.y += self.bird.velocity
        self.bird.velocity += self.gravity
        
        self.score += 2
        
        if self.visualize:
            self.draw()
            
        if self.score > 30000:
            return False
        
        return self.get_collision()
    
    def get_closest_pipe(self):
        min = self.width
        min_pipe = None
        for p in self.pipes:
            if self.bird.x <= p.x+p.width and p.x < min:
                min = p.x
                min_pipe = p
        return min_pipe
    
    def get_collision(self):
        if self.bird.y >= self.height-self.bird.diameter or self.bird.y < 0:
            return False
        
        pipe = self.get_closest_pipe()
        if pipe is None:
            return True
        pipe_top = [pipe.x, 0, pipe.width, pipe.y]
        pipe_bottom = [pipe.x, pipe.y + pipe.gap_size, pipe.width, self.height]
        
        overlapping = True
        # top side
        # If one rectangle is on left side of other
        if self.bird.x > pipe_top[0] + pipe_top[2] or self.bird.x + self.bird.diameter < pipe_top[0]:
            overlapping = False
     
        # If one rectangle is above other
        if self.bird.y > pipe_top[3]:
            overlapping = False
 
        if overlapping:
            return False
            
        overlapping = True
        # bottom side
        if self.bird.x > pipe_bottom[0] + pipe_bottom[2] or self.bird.x + self.bird.diameter < pipe_bottom[0]:
            overlapping = False
            
        # If one rectangle is above other
        if self.bird.y + self.bird.diameter < pipe_bottom[1]:
            overlapping = False
 
        return not overlapping
        
    def get_normalized_state(self):
        pipe = self.get_closest_pipe()
        if pipe is None:
            dx = self.width
            dy = self.height//2
        else:
            dx = pipe.x - (self.bird.x + self.bird.diameter)
            dy = (pipe.y + pipe.gap_size//2) - (self.bird.y + self.bird.diameter//2)
        dx = (dx + self.width)/(2*self.width)
        dy = (dy + self.height)/(2*self.height)
        return [dx,dy]
       
    def draw(self):
        # reset screen
        self.screen.fill(pygame.Color('black'), pygame.Rect(0,0,self.width,self.height))
        # draw bird
        pygame.draw.rect(self.screen, pygame.Color('white'), pygame.Rect(self.bird.x,self.bird.y,self.bird.diameter,self.bird.diameter))        
        # draw pipes
        for p in self.pipes:
            w = self.width-p.x if self.width-p.x < p.width else p.width
            pygame.draw.rect(self.screen, pygame.Color('white'), pygame.Rect(p.x, 0, w, p.y))
            pygame.draw.rect(self.screen, pygame.Color('white'), pygame.Rect(p.x, p.y + p.gap_size, w, self.height))
        # update screen
        pygame.display.flip()
        
    def get_action(self):
        events = pygame.event.get()
        for e in events:
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    return 1
        return 0
  

class Node: 
    
    def __init__(self,key,name,x=0,y=0):
        self.x = x
        self.y = y
        self.name = name
        self.key = key
        
class Edge:
    
    def __init__(self, key1, key2, enabled):
        self.key1 = key1
        self.key2 = key2
        self.enabled = enabled
        
  
def draw_net(screen, config, genome, name_list, width, height):
    screen.fill(pygame.Color('grey'), pygame.Rect(width,0,400,height))
    
    # get inputs
    inputs = {}
    for k in config.genome_config.input_keys:
        name = name_list.get(k, str(k))
        inputs[k] = Node(k,name)
        
    # get outputs 
    outputs = {}
    for k in config.genome_config.output_keys:
        name = name_list.get(k, str(k))
        outputs[k] = Node(k,name)
    
    # get hidden 
    nodes = set(genome.nodes.keys())
    hidden = {}
    for n in nodes:
        if n not in inputs and n not in outputs:
            hidden[n] = Node(n,str(n))
    
    offset = 20
    radius = 30
    # draw inputs
    start = len(inputs)//2
    for k,n in inputs.items():
        h = height//2 + start * (offset + 2*radius) if len(inputs)%2==1 else height//2 + start * (offset + 2*radius) - radius - offset//2
        pygame.draw.circle(screen, pygame.Color('white'), [width+50, h], radius)
        pygame.draw.circle(screen, pygame.Color('black'), [width+50, h], radius, 1)
        n.x = width+50
        n.y = h
        start -= 1
    
    # draw hidden
    start = len(hidden)//2
    for k,n in hidden.items():
        h = height//2 + start * (offset + 2*radius) if len(hidden)%2==1 else height//2 + start * (offset + 2*radius) - radius - offset//2
        pygame.draw.circle(screen, pygame.Color('white'), [width+200, h], radius)
        pygame.draw.circle(screen, pygame.Color('black'), [width+200, h], radius, 1)
        n.x = width+200
        n.y = h
        start -= 1
    
    # draw outputs
    start = len(outputs)//2
    for k,n in outputs.items():
        h = height//2 + start * (offset + 2*radius) if len(outputs)%2==1 else height//2 + start * (offset + 2*radius) - radius - offset//2
        pygame.draw.circle(screen, pygame.Color('white'), [width+350, h], radius)
        pygame.draw.circle(screen, pygame.Color('black'), [width+350, h], radius, 1)
        n.x = width+350
        n.y = h
        start -= 1
        
    # get edges
    all_nodes = {**inputs,**hidden,**outputs}
    for cg in genome.connections.values():
        input, output = cg.key
        color = pygame.Color('green') if cg.weight > 0 else pygame.Color('red')
        if cg.enabled:
            pygame.draw.line(screen, color, [all_nodes[input].x,all_nodes[input].y], [all_nodes[output].x,all_nodes[output].y])
        else:
            padlib.DashedLine(screen, color, pygame.Color('grey'), [all_nodes[input].x,all_nodes[input].y], [all_nodes[output].x,all_nodes[output].y])

    # draw names
    for k,n in all_nodes.items():
        font = pygame.font.SysFont('Helvetica', 12)
        text = font.render(n.name, False, pygame.Color('black'))
        screen.blit(text,[n.x-text.get_rect().width//2,n.y-text.get_rect().height//2])
    
def main():      
    width = 600
    height = 600
    
    game = Game(width,height,true)
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    with open('winner-feedforward', 'rb') as f:
        winner = pickle.load(f)
    
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    node_names = {-1: 'x', -2: 'y', 0: 'wait', 1: 'flap'}
    draw_net(game.screen, config, winner, node_names, width, height)
    
    running = True
    while running:
        inputs = game.get_normalized_state()
        action = net.activate(inputs)
        running = game.step(np.argmax(action))

     
    
  
if __name__ == '__main__':
    main()