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

# OPENING THE GAME

# create FIFO pipes
os.system("mkfifo ale_fifo_out")
os.system("mkfifo ale_fifo_in")

# launch ALE with appropriate commands in the background
os.system(
    './../libraries/ale/ale -game_controller fifo_named '
    '-run_length_encoding false -display_screen true ../libraries/ale/roms/breakout.bin &')

# open communication with pipes
fin = open('ale_fifo_out')
fout = open('ale_fifo_in', 'w')

# saves the image sizes (160*210) for breakout
size = fin.readline()[:-1].split("-")

# first thing we send to ALE is the output options- we want to get only image data (hence the zeros)
fout.write("1,0,0,0\n")
# probably we need fout.write("1,0,0,1\n") in order to understand "game over!"

# send the lines written to pipe
fout.flush()

# Playing lots of games
for i in range(numGames):

    # read what ALE responds- it should be the initial game screen
    fin.readline()

    # send the fist command
    # first command has to be 1,0 or 1,1, because the game starts not when you press "right" or "left",
    # but whn you press "fire"(space or enter or whatever)
    fout.write("1,1\n")
    fout.flush()

    # Playing many moves until Game Over (t=-1)
    t = 0
    while t != -1:
        # if game is over, then t = -1 otherwise t += 1:
        t = -1

fin.close()
fout.close()
