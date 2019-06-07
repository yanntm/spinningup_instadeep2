#!/usr/bin/env python

import unittest
from functools import partial

import gym
import tensorflow as tf

from spinup import ppo
from spinup import ddpg

class TestPPO(unittest.TestCase):
    def test_cartpole(self):
        ''' Test training a small agent in a simple environment '''
        env_fn = partial(gym.make, 'LunarLander-v2')
        ac_kwargs = dict(hidden_sizes=(32,))
        with tf.Graph().as_default():
            ppo(env_fn, steps_per_epoch=100, epochs=10, ac_kwargs=ac_kwargs)
            #ddpg(env_fn,steps_per_epoch=100,epochs=10,ac_kwargs=ac_kwargs)
        # TODO: ensure policy has got better at the task


if __name__ == '__main__':
    unittest.main()
