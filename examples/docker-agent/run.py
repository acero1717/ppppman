"""Implementation of a simple deterministic agent using Docker."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pommerman import agents
from pommerman.runner import DockerAgentRunner
from train import *

import random


def main():
    '''Inits and runs a Docker Agent'''
    agent = DQNAgent(DQN())
    agent.run()


if __name__ == "__main__":
    main()
