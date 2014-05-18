

import os
import numpy as np
from preprocessor import Preprocessor

class ALE:
    actions = [np.uint8(0), np.uint8(1), np.uint8(3), np.uint8(4)]
    current_reward = 0
    next_screen = ""
    game_over = False
    memory = ""
    skip_frames = 4
    display_screen = "true"
    game_ROM = '../libraries/ale/roms/breakout.bin'
    fin = ""
    fout = ""
    preprocessor = None
    
    def __init__(self,  memory, display_screen="true", skip_frames=4, game_ROM='../libraries/ale/roms/breakout.bin'):
        self.display_screen = display_screen
        self.skip_frames = skip_frames
        self.memory = memory
        
        #: create FIFO pipes
        os.system("mkfifo ale_fifo_out")
        os.system("mkfifo ale_fifo_in")

        #: launch ALE with appropriate commands in the background
        command='./../libraries/ale/ale -game_controller fifo_named -disable_colour_averaging -run_length_encoding false -frame_skip '+str(self.skip_frames)+' -display_screen '+self.display_screen+" "+self.game_ROM+" &"
        os.system(command)

        #: open communication with pipes
        self.fin = open('ale_fifo_out')
        self.fout = open('ale_fifo_in', 'w')
        
        input=self.fin.readline()[:-1]
        size = input.split("-")  # saves the image sizes (160*210) for breakout

        #: first thing we send to ALE is the output options- we want to get only image data and episode info(hence the zeros)
        self.fout.write("1,0,0,1\n")
        self.fout.flush()  # send the lines written to pipe

        # initialise preprocessor
        self.preprocessor = Preprocessor()


    def new_game(self):
        self.next_image, episode_info = self.fin.readline()[:-2].split(":")   # read from ALE: the initial game screen + episode info
        self.game_over = bool(int(episode_info.split(",")[0]))
        self.current_reward = int(episode_info.split(",")[1])

        #: preprocess the image and the image to memory D
        self.memory.add_first(self.preprocessor.process(self.next_image))
        
        
        #: send the fist command
        # first command has to be 1,0 or 1,1, because the game starts when you press "fire!",
        self.fout.write("1,0\n")
        self.fout.flush()
        self.fin.readline()
        
    def end_game(self):
        #: tell the memory that we lost
        self.memory.add_last()
        
        #: send reset command to ALE
        self.fout.write("45,45\n")
        self.fout.flush()
        self.game_over = False # just in case, but new_game should do it anyway
        
    def store_step(self, action):
        self.memory.add(action, self.current_reward, self.preprocessor.process(self.next_image))
    
    def move(self, action):
        """
        Sends action to ALE and reads responds
        """
        #: Write and send to ALE stuff
        self.fout.write(str(action)+",0\n")
        self.fout.flush()

        #: Read from ALE
        self.next_image, episode_info = self.fin.readline()[:-2].split(":")
        self.game_over = bool(int(episode_info.split(",")[0]))
        self.current_reward = int(episode_info.split(",")[1])