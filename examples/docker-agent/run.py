"""Implementation of a simple deterministic agent using Docker."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pommerman import agents
from pommerman.runner import DockerAgentRunner
from train import *

import random

class MyAgent(DockerAgentRunner):
    '''An example Docker agent class'''

    def __init__(self):
        self.model = DQN().to(device)
        import os
        if os.path.exists("./model_good.pth"):
            self.model = torch.load("model_good.pth", map_location='cpu')
        self._agent = agents.SimpleAgent()

    def act(self, observation, action_space):
        obs = self.translate_obs(observation)
        obs = torch.from_numpy(obs).float().to(self.device)
        self.obs_fps.append(obs)
        obs = torch.cat(self.obs_fps[-4:])
        sample = random.random()
        if sample > 0.1:
            re_action = self.model(obs).argmax().item()
            return re_action
        else:
            return self._agent.act(observation, action_space)

def main():
    '''Inits and runs a Docker Agent'''
    agent = MyAgent()
    agent.run()


if __name__ == "__main__":
    main()
