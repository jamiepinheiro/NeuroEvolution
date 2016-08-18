import pygame
from pygame.locals import *
from PIL import Image

from neat import nn, population

import random
import time
import math

###################################################################################################################################

FPS, WIDTH, HEIGHT = 60, 1920, 1080
generation = 0;
IMAGES = {}
obstacles = []
xTimer = 0
xOffset = 0.05

###################################################################################################################################

class Car:

    def __init__(self, genome):
        self.genome = genome
        self.nn = nn.create_feed_forward_phenotype(genome)
        self.alive = True
        self.speed = 2
        self.direction = 0
        self.rect = pygame.Rect(WIDTH/5 + 20, HEIGHT/2, 8, 8)

    def image(self):
        return pygame.transform.rotate(IMAGES["car"], -self.direction * 57.3 - 90)

    def decision(self):
        inputs = [1000, 1000] # [left, right]
        xDis = 1000;

        xRounded = round(self.rect.x/8) * 8
        yRounded = round(self.rect.y/8) * 8

        closestX1, closestX2 = 1000, 1000;
        closest1, closest2 = None, None

        for obstacle in obstacles:

            disX = abs(self.rect.x - obstacle.rect.x)

            obstacle.triggered = False

            if(disX < closestX1 and obstacle.rect.y < self.rect.y):
                if closest1 is not None:
                    closest1.triggered = False
                closestX1 = disX
                closest1 = obstacle
                closest1.triggered = True
            elif(disX < closestX2 and obstacle.rect.y > self.rect.y):
                if closest2 is not None:
                    closest2.triggered = False
                closestX2 = disX 
                closest2 = obstacle
                closest2.triggered = True

        inputs[0] = abs(closest1.rect.y - self.rect.y)
        inputs[1] = abs(closest2.rect.y - self.rect.y)

        outputs = self.nn.serial_activate(inputs)

        if(outputs[0] > 0.5):
            self.direction += 0.05
        if(outputs[0] < 0.5):
            self.direction -= 0.05
        

    def move(self):
        self.rect.x += math.ceil(self.speed * math.cos(self.direction))
        self.rect.y += math.ceil(self.speed * math.sin(self.direction))
       
    def inCollision(self, obstacles):

        for obstacle in obstacles:
            if(math.sqrt((self.rect.x + self.rect.width/2 - obstacle.rect.x + obstacle.rect.width/2) ** 2 + (self.rect.y + self.rect.height/2 - obstacle.rect.y + obstacle.rect.height/2) ** 2) < 15):
                return True

        return False

class Obstacle:

    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, 8, 8)
        self.triggered = False

    def image(self):
        if (self.triggered):
            return IMAGES["obstacle_triggered"]
        else:
            return IMAGES["obstacle"]


###################################################################################################################################

def eval_fitness(genomes):

    global generation, obstacles

    obstacles = []

    for y in range(12):
        obstacles.append(Obstacle(WIDTH/5, HEIGHT/2-48 + y * 8))

    min, max, chagneFreq = -5, 5, 2
    currentY = HEIGHT/2
    for x in range(1000):
        obstacles.append(Obstacle(WIDTH/5 + x * 8, currentY + 48))
        obstacles.append(Obstacle(WIDTH/5 + x * 8, currentY - 48))
        if(currentY < 150):
            currentY += 8
        elif(currentY > 930):
            currentY -= 8
        else:
            if(x % chagneFreq == 0):
                currentY += round(random.uniform(min, max))
            if(x % 20 == 0):
                min = round(random.uniform(-10, 0))
                min = round(random.uniform(0, 10))
                chagneFreq = round(random.uniform(1, 3))

    car = Car(genomes[0])
    car.rect.height = HEIGHT/2

    cars = []
    for genome in genomes:
        cars.append(Car(genome))

    cars_alive = len(cars)

    start = time.time()

    markersOffset = 0

    done = False

    while cars_alive and done == False:

        for event in pygame.event.get(): # User did something
            if event.type == pygame.QUIT: # If user clicked close
                done = True # Flag that we are done so we exit this loop
                break

        xTimer = (time.time() - start)

        SCREEN.fill(pygame.Color(255, 255, 255, 255))

        for obstacle in obstacles:
            if(xTimer > xOffset):
                obstacle.rect.x -= 2;
            SCREEN.blit(obstacle.image(), (obstacle.rect.x , obstacle.rect.y))

        for car in cars:
            if not car.alive:
                continue
            if car.inCollision(obstacles):
                car.alive = False
                cars_alive -= 1
                lifespan = float(time.time() - start)
                car.genome.fitness = lifespan/60 + math.sqrt((car.rect.x - WIDTH/5) ** 2 + (car.rect.y - HEIGHT/2) ** 2)/1000
            if(xTimer > xOffset):
                car.rect.x -= 2;
            car.decision()
            car.move()
            SCREEN.blit(car.image(), (car.rect.x , car.rect.y))

        if(xTimer > xOffset):
            markersOffset -= 2

        if(xTimer > xOffset):
            start = time.time()


        # print statistics

        label1 = FONT.render('ALIVE: ' + str(cars_alive), 2, (0,0,0))
        label2 = FONT.render('TIME: ' + str(time.time() - start), 2, (0,0,0))
        label3 = FONT.render('GENERATION: ' + str(generation), 2, (0,0,0))
        SCREEN.blit(label1, (WIDTH/2, 100))
        SCREEN.blit(label2, (WIDTH/2, 120))
        SCREEN.blit(label3, (WIDTH/2, 140))

        for x in range(10):
            label = FONT.render(str(x * 100) + "m", 10, (0,0,0))
            SCREEN.blit(label, (WIDTH/2 + x * 800 + markersOffset, 1000))

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
    IMAGES['obstacle_triggered']  = pygame.image.load("pics/obstacle_triggered.png").convert()

    pygame.display.flip()
    pygame.display.update()
    
    pop = population.Population('car_config')
    pop.run(eval_fitness, 10000)

###################################################################################################################################

if __name__ == '__main__':
    main()
