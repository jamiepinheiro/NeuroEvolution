import pygame
from pygame.locals import *
from PIL import Image

from neat import nn, population

import random as rnd
import time
import math

###################################################################################################################################

FPS, WIDTH, HEIGHT = 60, 1920, 1080
generation = 0;
IMAGES = {}
obstacles = []

###################################################################################################################################

class Car:

    def __init__(self, genome):
        self.genome = genome
        self.nn = nn.create_feed_forward_phenotype(genome)
        self.alive = True
        self.speed = 2
        self.direction = 0
        self.rect = pygame.Rect(WIDTH/2, HEIGHT - 120, 20, 20)

    def image(self):
        return pygame.transform.rotate(IMAGES["car"], self.direction * 57.3)

    def decision(self, x, y):
        inputs = [x/WIDTH, y/HEIGHT]
        outputs = self.nn.serial_activate(inputs)
        self.direction = (outputs[0] - 0.5) * 6.28 * 2

    def move(self):
        self.rect.x += self.speed * math.cos(self.direction)
        self.rect.y += self.speed * math.sin(self.direction)

    def inCollision(self, obstacles):

    		for obstacle in obstacles:
    			if(math.sqrt((self.rect.x + self.rect.width/2 - obstacle.rect.x + obstacle.rect.width/2) ** 2 + (self.rect.y + self.rect.height/2 - obstacle.rect.y + obstacle.rect.height/2) ** 2) < 8):
							return True
                else:
                        return False

class Obstacle:

		def __init__(self, x, y):
				self.rect = pygame.Rect(x, y, 8, 8)

		def image(self):
				return IMAGES["obstacle"]


###################################################################################################################################


def sigmoid(x):
  return 1 / (1 + math.exp(-x))


###################################################################################################################################

def eval_fitness(genomes):
    global  generation, highscore

    car = Car(genomes[0])
    car.rect.height = HEIGHT/2

    cars = []
    for genome in genomes:
        cars.append(Car(genome))


    cars_alive = len(cars)

    start = time.time()

    while cars_alive:

        SCREEN.fill(pygame.Color(255, 255, 255, 255))

        for obstacle in obstacles:
        	 SCREEN.blit(obstacle.image(), (obstacle.rect.x , obstacle.rect.y))

        for car in cars:
            if not car.alive:
                continue
            if car.inCollision(obstacles):
                car.alive = False
                cars_alive -= 1
                lifespan = float(time.time() - start)
                car.genome.fitness = sigmoid(math.sqrt(car.rect.x ** 2 + car.rect.y ** 2)/50 + lifespan/60)
            car.decision(-1, 1)
            car.move()
            SCREEN.blit(car.image(), (car.rect.x , car.rect.y))


        # print statistics

        label1 = FONT.render('ALIVE: ' + str(len(obstacles)), 2, (0,0,0))
        label2 = FONT.render('TIME: ' + str(time.time() - start), 2, (0,0,0))
        label3 = FONT.render('GENERATION: ' + str(generation), 2, (0,0,0))
        SCREEN.blit(label1, (WIDTH/2, 440))
        SCREEN.blit(label2, (WIDTH/2, 460))
        SCREEN.blit(label3, (WIDTH/2, 480))

        # update the screen and tick the clock
        pygame.display.update()
        pygame.display.flip()
        FPSCLOCK.tick(FPS)
    generation += 1

###################################################################################################################################

def main():

    global FPSCLOCK, SCREEN, FONT
    pygame.init()
    FONT = pygame.font.SysFont("arial", 15)
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Car NeuroEvolution")

    IMAGES['car'] = pygame.image.load("pics/car.png").convert()
    IMAGES['obstacle']  = pygame.image.load("pics/obstacle.png").convert()

    pygame.display.flip()
    pygame.display.update()
 
    # hold until space is pushed
    hold = True
    while hold:
        for event in pygame.event.get():
            if event.type == KEYDOWN and event.key == K_SPACE:
                hold = False

    im = Image.open("pics/map.png")
    mapLayout = im.load()

    for x in range(240):
    	for y in range(135):
    		if(mapLayout[x, y] != 0):
    			obstacles.append(Obstacle(x * 8, y * 8))


    
    pop = population.Population('car_config')
    pop.run(eval_fitness, 10000)



    

###################################################################################################################################

if __name__ == '__main__':
    main()
