import retro
import numpy as np
import cv2
import neat
import pickle
from meta_game_functions import game_loader

# Loading Sonic 1 with the stage we want to use the algorithm on
sonic1 = game_loader('SonicTheHedgehog-Genesis','GreenHillZone.Act1')

# Image array for cv
imgArr = []



def eval_genomes(genomes, config):

    for genome_id, genome in genomes:

        # environment print screen
        observation = sonic1.reset()

        # set environment shape and size to input in the neural network
        inx, iny, inc = sonic1.observation_space.shape

        # we take inx and iny, divide those by 8 then multiply them together
        # we also use this number to get the # of inputs used in config-feedforward

        inx = int(inx/8)
        iny = int(iny/8)

        # create the neural network
        sonicNeuralNetwork = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        # variables used in learning
        currentFitness = 0
        currentMaxFitness = 0
        frame = 0
        counter = 0
        xpos = 0
        xposMax = 0
        currentRingCount = 0
        defaultLives = 3

        done = False

        ####--- Control Variables ---####
        activatePositionReward = True  # activate reward by moving forward in x
        positionReward = 1  # reward amount for moving forward

        activateRingReward = True  # ring reward activation
        ringReward = 0.5  # a smaller reward for collecting rings. They are
                          # not important to completing the level, but can
                          # keep sonic alive and provides incentive

        deathWard = 5  # punishment for dying

        while not done:
            sonic1.render()
            frame +=1

            observation = cv2.resize(observation, (inx, iny))  # resize
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            observation = np.reshape(observation, (inx, iny))

            # compress 2d image into an array
            for i in observation:
                for j in i:
                    imgArr.append(j)

            nnOutput = sonicNeuralNetwork.activate(imgArr)

            observation, reward, done, info = sonic1.step(nnOutput)

            imgArr.clear()

            # get sonics position, rings and lives. Names come from retro emulator data JSON
            xpos = info['x']
            xposEnd = info['screen_x_end']
            ringCount = info['rings']
            lives = info['lives']

            ####--- Rewads and Punishments ---####

            # reward for moving forward
            if xpos > xposMax:
                currentFitness += positionReward
                xposMax = xpos

            # reward for collecting rings
            if ringCount > currentRingCount:
                currentFitness += ringReward
                currentRingCount = ringCount

            # light punishment for losing rings
            if ringCount < currentRingCount:
                currentFitness -= ringReward
                currentRingCount = ringCount

            # heavy punishment for dying to stage hazards
            if lives < defaultLives:
                deathtFitness = currentFitness - deathWard
                currentFitness = deathtFitness


            if currentFitness > currentMaxFitness:
                currentMaxFitness = currentFitness
                counter = 0
            else:
                counter +=1

            # set current fitness to max when the end of the screen is reached
            if xpos > xposEnd:
                currentFitness += 100000  # highest fitness threshold value in config file
                done = True

            # genome gets 250 attempts to move to the right 1 pixel
            if done or counter == 250:
                done = True
                print("Genome: ",genome_id,"|| Fitness: ", currentFitness, "|| Ring Count: ", currentRingCount)


            genome.fitness = currentFitness


config = neat.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,'config-feedforward.txt')

p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10)) # every 10 generations, create a checkpoint
                                      # useful for saving and restoring simulation states

winner = p.run(eval_genomes)

# pickling to save

with open('winner.pkl','wb') as output:
    pickle.dump(winner, output, 1)















