from utils import *
import time
from vgg19_model import VGG19
import tensorflow as tf
import numpy as np
from scipy import misc
import pickle

def main():
    with open('tmp.pkl', 'rb') as f:
        tmp = pickle.load(f)
    lr_arr = [0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6]
    for key in tmp.keys():
        if key == 'conv1_1' or key == 'conv2_2':
            reshaped_target = tmp[key].reshape([1]+list(tmp[key].shape))
            for lr in lr_arr:
                with tf.Graph().as_default():
                    recon_image_by_given_layer(reshaped_target, key, 200000, 10000, False, lr)

if __name__=="__main__":
    main()
    print ("Good good study, day day up!")
