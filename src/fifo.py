###############
# This little script shows how to initialize communication with ALE and
# send commands to move left and right in Breakout game.
# Author: Ardi Tampuu
###############

import os
#create FIFO pipes
os.system("mkfifo ale_fifo_out")
os.system("mkfifo ale_fifo_in")

#launch ALE with appropriate commands in the background
os.system("./ale -game_controller fifo_named -run_length_encoding true -display_screen true roms/breakout.bin &")


#oppen communication with pipes
fin = open('ale_fifo_out')
fout = open('ale_fifo_in', 'w')

size = fin.readline()[:-1].split("-")  # saves the image sizes (160*210 for breakout

#first thing we send to ALE is the output options- we want to get only image data (hence the zeros)
fout.write("1,0,0,0\n")
fout.flush()  # send the lines written to pipe
fin.readline()   # read what ALE responds- it should be the initial game screen

# send the fist command
# first command has to be 1,0 or 1,1, because the game starts not when you press "right" or "left",
# but whn you press "fire"(space or enter or whatever)
fout.write("1,1\n")
fout.flush()
img = fin.readline()[:-2]  # remove the ":\n" from the end of hexadecimal sequence


#now let's do some moving left, stopping and moving right:

for i in range(20):  # for twenty frames
    fout.write("4,0\n")  # move left
    fout.flush()  # send
    img = fin.readline()[:-2]  # read


fout.write("3,0\n")  # "go right" -> doing it only once stops moving left
fout.flush()
img = fin.readline()[:-2]

for i in range(20):  # for twenty frames
    fout.write("0,0\n")  # do nothing
    fout.flush()
    img = fin.readline()[:-2]
    
for i in range(20):  # for twenty frames
    fout.write("3,0\n")  # move right
    fout.flush()
    img = fin.readline()[:-2]

fout.write("4,0\n")  # "go left" -> doing it only once stops moving right
fout.flush()
img = fin.readline()[:-2]

for i in range(20):  # for twenty frames
    fout.write("0,0\n")  # do nothing
    fout.flush()
    img = fin.readline()[:-2]

for i in range(20):  # for twenty frames
    fout.write("4,0\n")  # move left
    fout.flush()
    img = fin.readline()[:-2]

fout.write("3,0\n")  # "go right" -> doing it only once stops moving left
fout.flush()
img = fin.readline()[:-2]

for i in range(20):  # for twenty frames
    fout.write("0,0\n")  # do nothing
    fout.flush()
    img = fin.readline()[:-2]
    
for i in range(20):  # for twenty frames
    fout.write("3,0\n")  # go right
    fout.flush()
    img = fin.readline()[:-2]
    
for i in range(20):  # for twenty frames
    fout.write("4,0\n")  # move left
    fout.flush()
    img = fin.readline()[:-2]

fout.write("3,0\n")  # "go right" -> doing it only once stops moving left
fout.flush()
img = fin.readline()[:-2]

for i in range(20):  # for twenty frames
    fout.write("0,0\n")  # do nothing
    fout.flush()
    img = fin.readline()[:-2]
    
for i in range(20):  # for twenty frames
    fout.write("3,0\n")  # do nothing
    fout.flush()
    img = fin.readline()[:-2]

#etc.....


fin.close()
fout.close()
