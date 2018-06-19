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

    for key in tmp.keys():
        reshaped_target = tmp[key].reshape([1]+list(tmp[key].shape))
        recon_image_by_given_layer(reshaped_target, key)

if __name__=="__main__":
    main()
    print ("Good good study, day day up!")
