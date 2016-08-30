import pygame
from neat import nn, population

import random as rnd
import time
import math

FPS, WIDTH, HEIGHT = 60, 2048, 512
BLACK = pygame.Color(0, 0, 0, 255)

generation = 0

class Car(pygame.sprite.Sprite):

    def __init__(self, genome):

        self.genome = genome
        self.nn = nn.create_feed_forward_phenotype(genome)
        self.direction = 0;
        self.speed = 0;

        self.image = pygame.Surface([20, 20])
        self.image.fill(pygame.Color(10, 255, 255, 255))
        self.rect = self.image.get_rect()

        self.rect.x = WIDTH/2
        self.rect.y = HEIGHT/2

        super(Car, self).__init__()

    def decision(self, x, y):
        inputs = [x/WIDTH, y/HEIGHT]
        outputs = self.nn.serial_activate(inputs)

        self.direction = (outputs[0] - 0.5) * 6.28

    def move(self):
        self.rect.x += self.speed * math.cos(self.direction)
        self.rect.y += self.speed * math.sin(self.direction)


def eval_fitness(genomes):
    global generation

    cars = pygame.sprite.Group()
    for genome in genomes:
        cars.add(Car(genome))

    cars_alive = len(cars)
    for car in cars:
        car.rect.x += 5 * rnd.randint(-5,5)
        car.direction += 2 * rnd.randint(-3, 3)
        car.speed += 10 * rnd.randint(-1,1)

    start = time.time()

    while time.time() - start < 3:

        positions = ""
        speeds = ""
        for car in cars:
            car.move()
            car.decision(car.rect.x, car.rect.y)
            positions += "[ " + str(car.rect.x) + ", " + str(car.rect.y) + " ], "
            speeds += str(car.direction) + ', '

        SCREEN.fill(pygame.Color(255, 255, 255, 255))
        cars.draw(SCREEN)
        
        label2 = FONT.render('TIME: ' + str(time.time() - start), 2, (0,0,0))
        label3 = FONT.render('GENERATION: ' + str(generation), 2, (0,0,0))
        SCREEN.blit(label2, (20, 460))
        SCREEN.blit(label3, (20, 480))

        #update screen
        pygame.display.flip()
        CLOCK.tick(FPS)
        pygame.display.update()

    for car in cars:
        car.genome.fitness = car.rect.x + car.rect.y

    print generation
    generation += 1


def main():

    global SCREEN, FONT, CLOCK

    pygame.init()
    FONT = pygame.font.SysFont("arial", 15)


    pygame.display.set_caption("Test")
    # Set the height and width of the screen
    SCREEN = pygame.display.set_mode([WIDTH, HEIGHT])

    # Loop until the user clicks the close button.
    done = False
     
    # Used to manage how fast the screen updates
    CLOCK = pygame.time.Clock()

    all_sprites_list = pygame.sprite.Group()
     
    # -------- Main Program Loop -----------
    while not done:
        # --- Main event loop
        for event in pygame.event.get(): # User did something
            if event.type == pygame.QUIT: # If user clicked close
                done = True # Flag that we are done so we exit this loop
     
        # --- Game logic

        pop = population.Population('test_config')
        pop.run(eval_fitness, 10000)

        #SCREEN.fill(pygame.Color(255, 255, 255, 255))

        #car = Car(0)
        #car.direction = math.sin(time.time()) * 3.14
        #car.speed = 5
        #all_sprites_list.add(car)

        #for car in all_sprites_list:
         #   car.move()

        #all_sprites_list.draw(SCREEN)

        #pygame.display.flip()
        #CLOCK.tick(FPS)
        #pygame.display.update()


if __name__ == '__main__':
    main()


