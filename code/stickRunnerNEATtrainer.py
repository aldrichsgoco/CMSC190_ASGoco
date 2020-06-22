import pygame
import random
import pickle
import numpy as np
import neat
import os
import visualize

from Sprites.OneByOne import OneByOne
from Sprites.OneByTwo import OneByTwo
from Sprites.OneByThree import OneByThree

from Sprites.TwoByOne import TwoByOne
from Sprites.TwoByTwo import TwoByTwo
from Sprites.TwoByThree import TwoByThree

from Sprites.ThreeByOne import ThreeByOne

from Sprites.Player import Player
import pickle

class Game(object):

    def __init__(self, playerArray, config):
        self.obstacleProbability = 0.01
        self.obstaclesOnScreen = []
        self.speed = 7.0
        self.lastQuotient = 0
        self.score = 0
        self.direction = -1
        self.running = True
        self.gameOver = False
        self.jumpSpeed = 7.2
        self.fontName = pygame.font.match_font('arial')
        self.clock = pygame.time.Clock()
        self.background_colour = (255,255,255)
        self.width = 900
        self.height = 600
        self.frameCount = 0
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.players = playerArray
        self.config = config
        self.totalPlayer = len(playerArray)
        self.alive = len(playerArray)
        
        #Generates the neutral network for the list of players
        for playerId, player in self.players:
            player.net = neat.nn.FeedForwardNetwork.create(player, config)
            player.fitness = 0
            player.alive = True
    

    # Display Text on Screen
    def drawText(self, text, size, x, y):
        font = pygame.font.Font(self.fontName, size)
        textSurface = font.render(text, True, (0, 0, 0))
        textRect = textSurface.get_rect()
        textRect.midtop = (x, y)
        self.screen.blit(textSurface, textRect)

    # Draw the game background
    def drawGameBackground(self):
        self.screen.fill(self.background_colour)
        pygame.draw.rect(self.screen, (0, 0, 0), (0, 550, 900, 1), 1)
    
    # Draw obstacles and players on screen
    def drawCharacter(self):
        
        for playerId, player in self.players:

            if player.alive:
                player.fitness += 0.001
                player.drawCharacter(self.screen, 0)
        
        for obstacles in self.obstaclesOnScreen:
            obstacles.drawCharacter(self.screen)
    
    #Draw the lines which represents the sensors for the input value in the neural network
    def drawLines(self):
        
        for playerId, player in self.players:

            if player.alive:
                if len(self.obstaclesOnScreen) > 0: #If there are obstacle
                    obstacleSize = self.getObstacleIndex(self.obstaclesOnScreen[0].__class__.__name__)
                    pygame.draw.line(self.screen, (255, 0, 0), (player.x + (41/2), player.y[0] + (40/2)),(self.obstaclesOnScreen[0].x + obstacleSize[0]/2, self.obstaclesOnScreen[0].y + obstacleSize[1]/2 ) , 3)
                    pygame.draw.line(self.screen, (0, 255, 0), (self.obstaclesOnScreen[0].x, self.obstaclesOnScreen[0].y - 10),(self.obstaclesOnScreen[0].x + obstacleSize[0] + 5 , self.obstaclesOnScreen[0].y - 10) , 5)
                    pygame.draw.line(self.screen, (0, 0, 255), (self.obstaclesOnScreen[0].x - 10, self.obstaclesOnScreen[0].y),(self.obstaclesOnScreen[0].x - 10, self.obstaclesOnScreen[0].y + obstacleSize[1] + 5 ) , 5)
    
    # Randomly generate game obstacles depending on obstacle probability
    def generateGameObstacles(self):
        if len(self.obstaclesOnScreen) == 0 or self.obstaclesOnScreen[len(self.obstaclesOnScreen) - 1].x < 600:
            if random.uniform(0,1) < self.obstacleProbability:
                obstacleNumber = random.randint(1,7)
                if obstacleNumber == 1:
                    self.obstaclesOnScreen.append(OneByOne(900, 515))
                elif obstacleNumber == 2:
                    self.obstaclesOnScreen.append(OneByTwo(900, 515))
                elif obstacleNumber == 3:
                    self.obstaclesOnScreen.append(OneByThree(900, 515))
                elif obstacleNumber == 4:
                    self.obstaclesOnScreen.append(TwoByOne(900, 500))
                elif obstacleNumber == 5:
                    self.obstaclesOnScreen.append(TwoByTwo(900, 500))
                elif obstacleNumber == 6:
                    self.obstaclesOnScreen.append(TwoByThree(900, 500))
                elif obstacleNumber == 7:
                    self.obstaclesOnScreen.append(ThreeByOne(900, 485))
                         
    # Kill players on collision
    def detectCollisionAndKillPlayer(self):
        for playerId, player in self.players:
            if player.detectCollision(self.obstaclesOnScreen[0]) and player.alive:
                self.alive -= 1
                player.alive = False

    #Gets the dimension of the obstacle 
    def getObstacleIndex(self, name):
        if name == "OneByOne":
            return (15, 30)

        if name == "OneByTwo":
            return (30, 30)

        if name == "OneByThree":
            return (45, 30)
            
        if name == "TwoByOne":
            return (15, 45)

        if name == "TwoByTwo":
            return (30, 45)

        if name == "TwoByThree":
            return (45, 45)
        
        if name == "ThreeByOne":
            return (15, 60)
        
    # Predict actions for all players which are alive
    def predictActionsForPlayers(self):
        for playerId, player in self.players:
            if player.alive:
                if len(self.obstaclesOnScreen) > 0: #If there are obstacle
                    obstacleNumber = self.getObstacleIndex(self.obstaclesOnScreen[0].__class__.__name__)
                    input = (float(obstacleNumber[0]),float(obstacleNumber[1]), float(self.obstaclesOnScreen[0].x - player.x))
                else: #If there are no obstacle        
                    input = (float(0),float(0), 0)

                output = player.net.activate(input)
                player.predictedAction = output[0]  
                    
    # Check if generation of Players are extinct
    def allDead(self):
        for playerId, player in self.players:
            if player.alive:
                return False
        self.gameOver = True
        return True

    #Makes all the players jump based on the output of the neural network
    def makePlayersJump(self):
        for playerId, player in self.players:
            if player.alive:
                if not player.isJumping:
                    if player.predictedAction > 0.5:
                        player.isJumping = True
                else: #calculates the jumping direction of the player
                    player.isJumping, player.direction = player.jump(player.isJumping, player.direction, self.jumpSpeed)

    #Removes the obstacle that has passed thru the screen and also propagates the obstacles themselves
    def cleanDeadObstaclesAndPropagate(self):
        
        for obstacle in self.obstaclesOnScreen:
            if obstacle.x > 30:
                break
            else: #If the obstacle has been bypassed
                self.score += 1

                for playerId, player in self.players:
                    if player.alive:
                        player.fitness += 1

                self.obstaclesOnScreen.pop(0)    

        for obstacle in self.obstaclesOnScreen:
            obstacle.propagate(self.speed)
    
    def game(self):
        pygame.init()
        pygame.display.set_caption('Stick Runner')
        self.drawGameBackground()
        pygame.display.flip()

        while self.running:
            self.clock.tick(0)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
            
            self.predictActionsForPlayers()

            self.makePlayersJump()
            
            self.drawGameBackground()
            self.generateGameObstacles()
            self.cleanDeadObstaclesAndPropagate()
            self.drawCharacter()
            self.drawLines()
            self.drawText('ALIVE: ' + str(self.alive) + ' OUT OF ' + str(self.totalPlayer), 30, 680, 80)
            self.drawText('SCORE: ' + str(self.score), 30, 700, 110)
            
            pygame.display.update()

            if len(self.obstaclesOnScreen) > 0:
                self.detectCollisionAndKillPlayer()
            
            if self.allDead():
                return
            
            if self.score == 51:
                return 
                     
class Simulate(object):
    def __init__(self):
        self.GENERATION_NUMBER = 0

    def main(self):
        #NEAT set up
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config')
        config = neat.Config(Player, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

        pop = neat.Population(config)
        pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)

        winner = pop.run(self.eval_genomes,30)

        print('\nBest genome:\n{!s}'.format(winner))

        #Visualizes the statistics involved in the training
        visualize.plot_stats(stats, ylog=False, view=True) 
        visualize.plot_species(stats, view=True)
        node_names = {-1:'Obstacle Width', -2: 'Obstacle Height', -3: 'Distance to Obstacle', 0:'Jump or Not'}
        visualize.draw_net(config, winner, True, node_names=node_names)
        
        # Save winner in a file
        with open('bestPlayer_better.pickle', 'wb') as handle:
            pickle.dump(winner, handle, protocol = pickle.HIGHEST_PROTOCOL)

    def eval_genomes(self, genomes, config):
        
        g = Game(genomes,config) 
        g.game()

sim = Simulate()
sim.main()