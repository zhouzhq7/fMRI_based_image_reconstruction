from utils import *
import time
from vgg19_model import VGG19
import tensorflow as tf
import numpy as np
from scipy import misc
import pickle

def reconstruct_images():
    with open('tmp.pkl', 'rb') as f:
        tmp = pickle.load(f)
    for key in tmp.keys():
        tmp[key] = tmp[key].reshape([1]+list(tmp[key].shape))
    targets = []
    for layer_name in LAYER_TO_BE_SAVED_LESS:
        targets.append(tmp[layer_name])

    all_layer = True
    if all_layer:
        with tf.Graph().as_default():
            recon_image_by_given_layer(targets, 'all_layers', 20000, 1000,
                                       use_prior=True, use_all_layers=all_layer)
    else:
        for key in tmp.keys():
            reshaped_target = tmp[key]
            with tf.Graph().as_default():
                recon_image_by_given_layer(reshaped_target, key, 200000, 10000,
                                           use_summary=False, lr=0.001, use_prior=True)

def main():
    data = read_images(print_not_found=True, image_id_file=TEST_IMAGE_ID_FILE,
                image_dir=TEST_IMAGE_DIR)
    pass

if __name__=="__main__":
    main()
    print ("Good good study, day day up!")
