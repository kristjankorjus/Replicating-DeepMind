###############
# This little script shows how to initialize communication with ALE and
# send commands to move left and right in Breakout game.
# Author: Ardi Tampuu
###############

import os
import re #regular expressions
import random
import time


def findBall(pixs):
    m=re.search("(0606|0000)(4\w){2,5}(0606|0000)", pixs)
    
    x, y=0, 0
    if m is not None:
        #print pixs[-10+m.start():10+m.start() ]
        beg=m.start()+6
        x, y= (beg/2)%160, (beg/2)/160
   
    else:
        print "Ball is gone and we need to wait"
        time.sleep(0.5)
        x, y=0, 0
    
    #print "ball location is; ", x, y
    return x, y

def findPaddle(pixs):
    m=re.search("(B6|00)(4\w){10,23}(46|00)", pixs)
    #print m
    x, y=0, 0
    if m is not None:
        center=m.start()+20
        #print pixs[-4+m.start():60+m.start() ]
        x, y= (center/2)%160, (center/2)/160
    
    else:
        print "Paddle has mysteriously disappeared"
        print pixs[189*160*2:190*160*2]
        time.sleep(3)
    

    #print "paddle location is:", x, y
    
    return x, y

def chooseAction(xs, ys,  paddleX,  paddleY, lastAction,  stopped):
    action="0,0\n"
   

    if stopped and (not xs<(paddleX-6)) and (not xs>(paddleX+6)):
        #print "I got ahead of the ball, stopped and am waiting now",  stopped
        action="0,0\n"
        
    elif stopped and xs<(paddleX-2):
        action="4,0\n"
        stopped=False
   
    elif stopped and xs>(paddleX+2):
        action="3,0\n"
        stopped=False

    elif not stopped and lastAction=="3,0\n": #if we went RIGHT last time
        if xs>=(paddleX): # if the ball is still RIGHT of paddle
            action="3,0\n"
        elif xs<(paddleX): #if we have passed the ball already, then stop
            action="4,0\n"
            stopped=True
        else:
            print "bug1"
            

    elif not stopped and lastAction=="4,0\n": #if we went LEFT last time
        if xs<=(paddleX): # if the ball is still left of paddle
            action="4,0\n"
        elif xs>(paddleX): #if we have passed the ball already, then stop
            action="3,0\n"
            stopped=True
        else:
            print "bug2"
            
    else:
        print "sth is reall really fishy here"

    
    
    #print "chose action: ", action
    #print "stiopped", stopped
    return action,  stopped
    



#create FIFO pipes
os.system("mkfifo ale_fifo_out")
os.system("mkfifo ale_fifo_in")

#launch ALE with appropriate commands in the background
os.system(
    './../libraries/ale/ale -game_controller fifo_named -run_length_encoding false -frame_skip 4 -display_screen true ../libraries/ale/roms/breakout.bin &')



#oppen communication with pipes
fin = open('ale_fifo_out')
fout = open('ale_fifo_in', 'w')


input=fin.readline()
print input
size = input[:-1].split("-")  # saves the image sizes (160*210) for breakout


#first thing we send to ALE is the output options- we want to get only image data (hence the zeros)
fout.write("1,0,0,1\n")
# probably we need fout.write("1,0,0,1\n") in order to understand "game over!"

fout.flush()  # send the lines written to pipe
input=fin.readline()   # read what ALE responds- it should be the initial game screen
print "length of first readline is:  ",  len(input)
image=input.split(":")[0]
print "length of image is", len(image)
print input.split(":")[1]
# send the fist command
# first command has to be 1,0 or 1,1, because the game starts not when you press "right" or "left",
# but whn you press "fire"(space or enter or whatever)
fout.write("1,1\n")
fout.flush()




#now let's do some moving, first define some useful variables:
bx, by=0, 0#ball location
px, py=0, 0 #paddle location
lastAction="4,0\n" #doesnt matter
stopped=False #if the paddle was told to stop moving last timestep

cumulated_reward=0
real_cumulated_reward=0

for i in range (5000):
    img,  episode_info= fin.readline()[:-2].split(":")
    episode_info=episode_info.split(",")

    reward=float(episode_info[1]) #reward obtained this timestep, NOT cumulated reward
    game_over=int(episode_info[0]) #sends 1 if episode ended this timestep


    if game_over==1:
        print "ALE tells us the episode has ended, that means we lost all our lives :("
        time.sleep(5.0) #lets take a break, so we can realize that this happened (in case it happens in the wrong place)
        #let's do something about it!!!
    
    if reward>0:
        real_cumulated_reward+=reward
        #article says all positive rewards are considered equal to 1
        reward=1
        cumulated_reward+=1
        print "got some points! The cumulated scores so far are: ", real_cumulated_reward,"   ->",  cumulated_reward
        time.sleep(0.5)
        #now we have to use this reward to update the network (to somehow update the Q-value of the last (state, action) pair   -- Q=reward+max(Q|a) )

    x, y=findBall(img) #sends the ball coordinates, if ball was not found, sends "0,0"
    
    
    pxt, pyt=findPaddle(img)
    if not pxt==158: #if the algorithm fails to detect paddle, it gives us the coordinates of the red square on the right (x=158) (which is of the same colour)
        px, py=pxt, pyt #if paddle was correctly found, update
    
    
    if not(x==0 and y==0): #if ball was found
        bx, by= x, y #update ball location, (otherwise use the pervious location)
        a, stopped=chooseAction(bx, by, px, py, lastAction, stopped)
        lastAction=a
    else: 
        #print "ball is gone, SHOOT!"
        a="1,1\n"


    
    #time.sleep(0.5)
    fout.write(a)
    fout.flush()


fin.close()
fout.close()

