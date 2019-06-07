#!/usr/bin/env python

import unittest
from functools import partial
import importlib
import gym
import tensorflow as tf

from spinup import ppo_pyco
#from spinup import ddpg

from pycolab.engine import Engine
from gym_pyco.envs import PycoEnv
from gym import Env

from functools import partial

def wrapPyco ( game : Engine)  -> Env :
    return PycoEnv(game)

def test_pyco():
    ''' Test training a small agent in a simple environment '''
    game_name = 'fluvial_natation-v0'
    mg = importlib.import_module('pycolab.examples.' + game_name )

    game = partial(wrapPyco, mg.make_game())

    ac_kwargs = dict(hidden_sizes=(32,))
    with tf.Graph().as_default():
        ppo_pyco(game, steps_per_epoch=100, epochs=1000, ac_kwargs=ac_kwargs)
    # TODO: ensure policy has got better at the task

test_pyco()