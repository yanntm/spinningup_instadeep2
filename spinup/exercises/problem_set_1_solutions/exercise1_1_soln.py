import tensorflow as tf
import numpy as np

EPS=1e-8

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi)) #size (32,10)
    return tf.reduce_sum(pre_sum, axis=1)

#pdf(x; mu, sigma) = exp(-0.5 (x - mu)**2 / sigma**2) / Z
#Z = (2 pi sigma**2)**0.5

