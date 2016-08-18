import pygame
from pygame.locals import *
from PIL import Image

from neat import nn, population, statistics

import random
import time
import math

###################################################################################################################################

FPS, WIDTH, HEIGHT = 60, 1920, 1080
generation = 0;
IMAGES = {}
obstacles = []
xTimer = 0
xOffset = 0

###################################################################################################################################

class Car:

    def __init__(self, genome):
        self.x = WIDTH/5 + 20
        self.y = HEIGHT/2
        self.genome = genome
        self.nn = nn.create_feed_forward_phenotype(genome)
        self.alive = True
        self.speed = 2
        self.direction = 0
        self.rect = pygame.Rect(self.x + xOffset, self.y, 8, 8)

    def image(self):
        self.rect = pygame.Rect(self.x + xOffset, self.y, 8, 8)
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

        if(closest1 is not None and closest2 is not None):
            inputs[0] = abs(closest1.rect.y - self.rect.y)
            inputs[1] = abs(closest2.rect.y - self.rect.y)

        outputs = self.nn.serial_activate(inputs)

        if(outputs[0] > 0.5):
            self.direction += 0.05
        if(outputs[0] < 0.5):
            self.direction -= 0.05
        

    def move(self):
        self.x += math.ceil(self.speed * math.cos(self.direction))
        self.y += math.ceil(self.speed * math.sin(self.direction))
       
    def inCollision(self, obstacles):

        for obstacle in obstacles:
            if(math.sqrt((self.rect.x + self.rect.width/2 - obstacle.rect.x + obstacle.rect.width/2) ** 2 + (self.rect.y + self.rect.height/2 - obstacle.rect.y + obstacle.rect.height/2) ** 2) < 15):
                return True

        return False

class Obstacle:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.rect = pygame.Rect(self.x + xOffset, self.y, 8, 8)
        self.triggered = False

    def image(self):
        if (self.triggered):
            return IMAGES["obstacle_triggered"]
        else:
            return IMAGES["obstacle"]

    def offset(self):
        self.rect = pygame.Rect(self.x + xOffset, self.y, 8, 8)


###################################################################################################################################

def eval_fitness(genomes):

    global generation, obstacles, xOffset

    obstacles = []

    

    min, max, changeFreq, j = -5, 5, 2, 0
    currentY = HEIGHT/2
    for x in range(1000):
        if(x == 0):
            for y in range(135):
                obstacles.append(Obstacle(WIDTH/5, y * 8))
        if(x == 999):
            for y in range(135):
                obstacles.append(Obstacle(WIDTH/5 + x * 8, y * 8))
        obstacles.append(Obstacle(WIDTH/5 + x * 8, currentY + 48))
        obstacles.append(Obstacle(WIDTH/5 + x * 8, currentY - 48))
        if(currentY < 150):
            min = 0
            max = 10
        elif(currentY > 930):
            min = -10
            max = 0
        else:
            if(x % changeFreq == 0):
                currentY += round(random.uniform(min, max))
            if(x % 20 == 0):
                min = round(random.uniform(-10, j))
                max = round(random.uniform(j, 10))
                changeFreq = round(random.uniform(2, 4))
            if(x % 50 == 0 or x == 0):
                j = round(random.uniform(-5, 5))

    car = Car(genomes[0])
    car.rect.height = HEIGHT/2

    cars = []
    for genome in genomes:
        cars.append(Car(genome))

    cars_alive = len(cars)

    offsetStart = time.time()
    lifespanStart = time.time()

    done = False
    xOffset = 0
    offsetTime = 0.02

    while cars_alive and done == False:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True 
                break

        xTimer = (time.time() - offsetStart)

        SCREEN.fill(pygame.Color(255, 255, 255, 255))

        for obstacle in obstacles:
            obstacle.offset()
            SCREEN.blit(obstacle.image(), (obstacle.rect.x , obstacle.rect.y))

        for car in cars:
            if not car.alive:
                continue
            if car.inCollision(obstacles):
                car.alive = False
                cars_alive -= 1
                lifespan = float(time.time() - lifespanStart)
                car.genome.fitness = lifespan/60 + car.x/500
            car.decision()
            car.move()
            SCREEN.blit(car.image(), (car.rect.x , car.rect.y))

        if(xTimer > offsetTime):
            for car in cars:
                if car.rect.x > WIDTH/2:
                    offsetTime -= 0.01

            if(len(cars) != 1):
                xOffset -= 2
                offsetStart = time.time()

        # print statistics
        label1 = FONT.render('ALIVE: ' + str(cars_alive), 2, (0,0,0))
        label2 = FONT.render('TIME: ' + str(time.time() - lifespanStart), 2, (0,0,0))
        label3 = FONT.render('GENERATION: ' + str(generation), 2, (0,0,0))
        SCREEN.blit(label1, (WIDTH/2, 100))
        SCREEN.blit(label2, (WIDTH/2, 120))
        SCREEN.blit(label3, (WIDTH/2, 140))


        for x in range(10):
            label = FONT.render(str(x * 100) + "m", 10, (0,0,0))
            SCREEN.blit(label, (WIDTH/2 + x * 800 + xOffset, 1000))

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

    statistics.save_stats(pop.statistics)
    statistics.save_species_count(pop.statistics)
    statistics.save_species_fitness(pop.statistics)

###################################################################################################################################

if __name__ == '__main__':
    main()
