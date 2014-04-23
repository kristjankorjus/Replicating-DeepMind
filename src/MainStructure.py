# The plan is to start building the structure of the algorithm (see algorithm.m or the paper)

###########
# imports #
###########
import os

##############
# PARAMETERS #
##############

# Number of games played
numGames = 2

##################
# MAIN ALGORITHM #
##################

# Playing lots of games
for i in range(numGames):

    # OPENING THE GAME

    # create FIFO pipes
    os.system("mkfifo ale_fifo_out")
    os.system("mkfifo ale_fifo_in")

    # launch ALE with appropriate commands in the background
    os.system(
        './../libraries/ale/ale -game_controller fifo_named '
        '-run_length_encoding false -display_screen true ../libraries/ale/roms/breakout.bin &')

    #oppen communication with pipes
    fin = open('ale_fifo_out')
    fout = open('ale_fifo_in', 'w')

    size = fin.readline()[:-1].split("-")  # saves the image sizes (160*210) for breakout

    # Playing many moves until Game Over (t=-1)
    t = 0
    while t != -1:
        # if game is over, then t = -1 otherwise t += 1:
        t = -1