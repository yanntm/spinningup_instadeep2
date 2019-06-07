import numpy as np
from functools import partial
from pycolab.engine import Engine
from gym_pyco.envs import PycoEnv
import importlib
from gym import Env



def wrapPyco(game: Engine) -> Env:
    return PycoEnv(game)

def random_agent(env_name,test=100):



    game_name = env_name
    mg = importlib.import_module('pycolab.examples.' + game_name)
    if env_name == 'warehouse_manager-v0':
        game = partial(wrapPyco, mg.make_game(level=0))
    else:
        game = partial(wrapPyco, mg.make_game())
    env = game()
    #This code is specific for pycolab

    act_dim = env.action_space.n
    mean_reward=[]
    for i in range(test):
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        while d == False :
            action = np.random.randint(act_dim)
            o, r, d, _ = env.step(action)
        mean_reward = mean_reward + [r]

    return np.mean(mean_reward)

random_agent('fluvial_natation-v1',test=1000)