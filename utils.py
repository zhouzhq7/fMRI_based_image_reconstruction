import time
import os
import csv
from scipy import misc
import numpy as np


DATA_DIR = "./data"
TRAINING_IMAGE_ID_FILE = "imageID_training.csv"
TRAIN_IMAGE_DIR = "./data/train"
VGG19_INPUT_IMAGE_SIZE = (224, 224, 3)

RESULT_DIR = "./results"

def read_images(print_not_found=False):
    # Read images ids from csv file
    print ("Start to load images...")

    start_time = time.time()

    imageid_file = os.path.join(DATA_DIR, TRAINING_IMAGE_ID_FILE)
    with open(imageid_file, 'r') as f:
        image_ids = list(csv.reader(f, delimiter=','))

    new_image_id_and_category_id = [[], []]
    rescaled_images = []
    cnt_not_found = 0
    for img_id in image_ids:

        cur_img_path = os.path.join(TRAIN_IMAGE_DIR, img_id[1])

        # Deal with the situation where image cannot be found
        if not os.path.isfile(cur_img_path):
            if print_not_found:
                print (img_id[1]+" is not found.")
                cnt_not_found += 1
            continue

        img = misc.imread(cur_img_path, mode="RGB")

        if img.shape != VGG19_INPUT_IMAGE_SIZE:
            img = misc.imresize(img, VGG19_INPUT_IMAGE_SIZE)

        if img.shape != VGG19_INPUT_IMAGE_SIZE:
            raise Exception("Image size unmatches.")
        # Rescale pixel value between 0 and 1
        img_rescaled = img/255.0

        # get image id
        new_image_id_and_category_id[0].append(float(img_id[0]))
        # get category id
        new_image_id_and_category_id[1].append(int(float(img_id[0])))

        rescaled_images.append(img_rescaled)

    if print_not_found:
        print ("{} images not found.".format(cnt_not_found))

    print ("Images has been loaded, takes {}s".format(int(time.time()-start_time)))

    return {"image_ids": new_image_id_and_category_id,
            "rescaled_images": np.array(rescaled_images)}

def save_dnn_feature_map(features):
   pass 
