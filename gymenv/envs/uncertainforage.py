import time
import gym
from gym import spaces
import numpy as np
from typing import Optional
#from gym.utils.renderer import Renderer

class UncertainForageEnv(gym.Env):
    #metadata = {"render_modes} currently not supported

    def __init__(self, render_mode: Optional[str] = None, Ambiguity: int = 0, Travel_time: int = 4,htime:float = 0.8):
        #assert render_mode is None or render_mode in self.metadata["render_modes"]
        
        self.Ambiguity = Ambiguity
        self.Travel_time = Travel_time
        self.step_counter = 0
        self.change = 0
        self.Probability = 0
        self.start_time = 0
        self.curr_time = 0
        self._curr_value = 0
        self._init_reward = 0
        self.block_duration = 300
        self.htime = htime
        self.done = 0

        self.observation_space = spaces.Dict(
            {
                "Initial reward":spaces.Discrete(100),
                "Probability":spaces.Box(low=0.0, high=1.0, shape=(1,)),
                "Time":spaces.Box(low=0.0, high=1200.0,shape=(1,)),
                "Current value":spaces.Discrete(100),
                "Patch change":spaces.Discrete(2),
            }
        )

        #We have two actions, harvest the patch or leave the patch
        self.action_space = spaces.Discrete(2)
    
    def _get_obs(self):
        return {"Initial reward" : self._init_reward, "Probability" : self.Probability, "Time": self.curr_time , "Current value":self._curr_value, "Patch change" : self.change}
    
    def _get_info(self):
        return {"Ambiguity" : self.Ambiguity, "Travel time": self.Travel_time}

    def reset(self, seed=None, return_info = False, options = None):
        super().reset(seed=seed)

        self.change = 0

        if np.random.random() > 0.5:
            self._init_reward = np.random.randint(90,100,1)[0]
            self.Probability = np.random.choice([0.2,0.3,0.5])
        else:
            self._init_reward = np.random.randint(45,55,1)[0]
            self.Probability = np.random.choice([0.5,0.6,0.8])

        self._curr_value = self._init_reward

        #self.renderer.reset()
        #self.renderer.render_step()
        
        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def step(self, action):
        if self.step_counter == 0:
            self.start_time = time.time()

        self.step_counter = self.step_counter + 1

        if action == 0:
            #handling time
            self.block_duration = self.block_duration - self.htime
            self.curr_time = self.curr_time + self.htime

            #reward calculation and patch value updation
            
            chance = np.random.random()
            if chance < self.Probability:
                reward = self._curr_value
            else:
                reward = 0
            
            if reward != 0:
                self._curr_value = int((np.random.random() * 0.1 + 0.85) * self._curr_value)
            
            if self._curr_value <= 5:
                self._curr_value = 0
            self.change = 0

        else:
            if np.random.random() > 0.5:
                self._init_reward = np.random.randint(90,100,1)[0]
                self.Probability = np.random.choice([0.2,0.3,0.5])
            else:
                self._init_reward = np.random.randint(45,55,1)[0]
                self.Probability = np.random.choice([0.5,0.6,0.8])

            self._curr_value = self._init_reward
            self.change = 1
            
            #Time penalty for travelling between patches
            self.block_duration = self.block_duration - self.Travel_time
            self.curr_time = self.curr_time + self.Travel_time

            reward = 0

        #Include time checks to end game
        if (time.time() - self.start_time >= self.block_duration):
            self.done = 1
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, self.done, info
    
    def close(self):
        return
