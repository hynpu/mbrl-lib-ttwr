import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import omegaconf

import mbrl.env. ttwr_steering as ttwr_env
import mbrl.env.cartpole_continuous as cartpole_env
import mbrl.env.reward_fns as reward_fns
import mbrl.env.termination_fns as termination_fns
import mbrl.models as models
import mbrl.planning as planning
import mbrl.util.common as common_util
import mbrl.util as util

seed = 0
env = ttwr_env.TtwrEnv(render_mode="human")
env.reset(seed)

# loop 100 episodes
obs = env.reset(seed)
# loop 200 steps
for _ in range(200):
    # get the action space
    act = env.action_space.sample()
    # get the next observation, reward, done, info
    obs, reward, done, _, info = env.step(act)
    # if done, break
    if done:
        break
    # render the environment
    env.render()
# close the environment
env.close()
