import retro
import numpy as np
import cv2
import neat
import pickle
from meta_game_functions import game_loader

# Loading Sonic 1 with the stage we want to use the algorithm on
env = game_loader('SonicTheHedgehog2-Genesis','EmeraldHillZone.Act2')

checkpoint_prefix = "checkpoints/EHZ2_2-checkpoint-"

# Image array for cv
imgArr = []



def eval_genomes(genomes, config):

    for genome_id, genome in genomes:

        # environment print screen
        observation = env.reset()

        # set environment shape and size to input in the neural network
        inx, iny, inc = env.observation_space.shape

        # we take inx and iny, divide those by 8 then multiply them together
        # we also use this number to get the # of inputs used in config-feedforward

        inx = int(inx/8)
        iny = int(iny/8)

        # create the neural network
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

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

        deathWard = 0.6  # punishment for dying

        while not done:
            env.render()
            frame +=1

            observation = cv2.resize(observation, (inx, iny))  # resize
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            observation = np.reshape(observation, (inx, iny))

            # compress 2d image into an array
            for i in observation:
                for j in i:
                    imgArr.append(j)

            nnOutput = net.activate(imgArr)

            # Grab the max ovalue from the nnOutput
            max_index = np.argmax([nnOutput])
            """
            These are the button options on the genesis
            Underneath are the 8 button combinations that are valuable in terms of progression
            Forcing the NN to predict one specific combination of buttons rather than a random group of them reduces the options it'll try from ~4096 to 8 which should speed up training.
            "buttons": 
            ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
             "relevant combinations":
            ( {{}, {LEFT}, {RIGHT}, {LEFT, DOWN}, {RIGHT, DOWN}, {DOWN}, {B}} ).
            """
            if max_index == 0:
                # Nothing
                mod_step_control = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif max_index == 1:
                # Left
                mod_step_control = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
            elif max_index == 2:
                # Right
                mod_step_control = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            elif max_index == 3:
                # Down left
                mod_step_control = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
            elif max_index == 4:
                # Right down
                mod_step_control = [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
            elif max_index == 5:
                # Down
                mod_step_control = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            elif max_index == 6:
                # Down B
                mod_step_control = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            elif max_index == 7:
                # B
                mod_step_control = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            else:
                print("max index out of range. No action.")
                mod_step_control = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            observation, reward, done, info = env.step(mod_step_control)

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
                currentFitness += 200000  # highest fitness threshold value in config file
                done = True

            # genome gets 250 attempts to move to the right 1 pixel
            if done or counter == 400:
                done = True
                print("Genome: ",genome_id,"|| Fitness: ", currentFitness, "|| Ring Count: ", currentRingCount)


            genome.fitness = currentFitness


config = neat.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,'config-feedforward.txt')

p = neat.Population(config)
p = neat.Checkpointer.restore_checkpoint('checkpoints/neat-checkpoint-7')
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10, filename_prefix=checkpoint_prefix)) # every 10 generations, create a checkpoint
                                      # useful for saving and restoring simulation states

winner = p.run(eval_genomes)

# pickling to save

with open('winner2.pkl','wb') as output:
    pickle.dump(winner, output, 1)















