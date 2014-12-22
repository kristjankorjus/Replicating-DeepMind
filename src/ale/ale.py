"""
ALE class launches the ALE game and manages the communication with it
"""

import os
import numpy as np
from preprocessor import Preprocessor
import traceback
import random

class ALE:
    actions = [np.uint8(0), np.uint8(1), np.uint8(3), np.uint8(4), np.uint8(11), np.uint8(12)]
    current_points = 0
    next_screen = ""
    game_over = False
    skip_frames = None
    display_screen = "true"
    game_ROM = None
    fin = ""
    fout = ""
    preprocessor = None
    
    def __init__(self, display_screen, skip_frames, game_ROM):
        """
        Initialize ALE class. Creates the FIFO pipes, launches ./ale and does the "handshake" phase of communication

        @param display_screen: bool, whether to show the game on screen or not
        @param skip_frames: int, number of frames to skip in the game emulator
        @param game_ROM: location of the game binary to launch with ./ale
        """

        self.display_screen = display_screen
        self.skip_frames = skip_frames
        self.game_ROM = game_ROM

        #: create FIFO pipes
        os.system("mkfifo ale_fifo_out")
        os.system("mkfifo ale_fifo_in")

        #: launch ALE with appropriate commands in the background
        command='./../libraries/ale/ale -max_num_episodes 0 -game_controller fifo_named -disable_colour_averaging true -run_length_encoding false -frame_skip '+str(self.skip_frames)+' -display_screen '+self.display_screen+" "+self.game_ROM+" &"
        os.system(command)

        #: open communication with pipes
        self.fin = open('ale_fifo_out')
        self.fout = open('ale_fifo_in', 'w')
        
        input = self.fin.readline()[:-1]
        size = input.split("-")  # saves the image sizes (160*210) for breakout

        #: first thing we send to ALE is the output options- we want to get only image data
        # and episode info(hence the zeros)
        self.fout.write("1,0,0,1\n")
        self.fout.flush()  # send the lines written to pipe

        #: initialize the variables that we will start receiving from ./ale
        self.next_image = []
        self.game_over = True
        self.current_points = 0

        #: initialise preprocessor
        self.preprocessor = Preprocessor()

    def new_game(self):
        """
        Start a new game when all lives are lost.
        """

        #: read from ALE:  game screen + episode info
        self.next_image, episode_info = self.fin.readline()[:-2].split(":")
        self.game_over = bool(int(episode_info.split(",")[0]))
        self.current_points = int(episode_info.split(",")[1])

        #: send the fist command
        #  first command has to be 1,0 or 1,1, because the game starts when you press "fire!",
        self.fout.write("1,0\n")
        self.fout.flush()
        self.fin.readline()

        #: preprocess the image and add the image to memory D using a special add function
        #self.memory.add_first(self.preprocessor.process(self.next_image))
        return self.preprocessor.process(self.next_image)

    def end_game(self):
        """
        When all lives are lost, end_game adds last frame to memory resets the system
        """
        #: tell the memory that we lost
        # self.memory.add_last() # this will be done in Main.py
        
        #: send reset command to ALE
        self.fout.write("45,45\n")
        self.fout.flush()
        self.game_over = False  # just in case, but new_game should do it anyway

    
    def move(self, action_index):
        """
        Sends action to ALE and reads responds
        @param action_index: int, the index of the chosen action in the list of available actions
        """
        #: Convert index to action
        action = self.actions[action_index]

	#: Generate a random number for the action of player B
        action_b = random.choice(range(255))


        #: Write and send to ALE stuff
        self.fout.write(str(action)+","+str(action_b)+"\n")
        #print "sent action to ALE: ",  str(action)+",0"
        self.fout.flush()

        #: Read from ALE
        line = self.fin.readline()
        try:
            self.next_image, episode_info = line[:-2].split(":")
            #print "got correct info from ALE: image + ", episode_info
        except:
            print "got an error in reading stuff from ALE"
            traceback.print_exc()
            print line
            exit()
        self.game_over = bool(int(episode_info.split(",")[0]))
        self.current_points = int(episode_info.split(",")[1])
        return self.current_points, self.preprocessor.process(self.next_image)
