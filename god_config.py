import os

DATA_DIR = "./data"
TRAINING_IMAGE_ID_FILE = "imageID_training.csv"
TRAIN_IMAGE_DIR = "./data/train"
VGG19_INPUT_IMAGE_SIZE = (224, 224, 3)

RESULT_DIR = "./results"
IMAGE_FEATURES_FILE_NAME_FULL = "images_feature.h5"
IMAGE_FEATURES_FILE_NAME_LESS = "images_feature_less.h5"
LAYER_TO_BE_SAVED_FULL = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
                     'conv3_1', 'conv3_2','conv3_3', 'conv3_4',
                     'conv4_1', 'conv4_2','conv4_3', 'conv4_4',
                     'conv5_1', 'conv5_2','conv5_3', 'conv5_4',
                     'fc6', 'fc7', 'fc8']

LAYER_TO_BE_SAVED_LESS = ['conv1_2', 'conv2_2',
                     'conv3_4',
                     'conv4_4',
                     'conv5_4',
                     'fc6', 'fc7', 'fc8']
LAYER_TO_BE_SAVED = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
                     'conv3_1', 'conv3_2','conv3_3', 'conv3_4',
                     'conv4_1', 'conv4_2','conv4_3', 'conv4_4',
                     'conv5_1', 'conv5_2','conv5_3', 'conv5_4',
                     'fc6', 'fc7', 'fc8']
IMAGE_ID_AND_CATEGORY_ID = ['image_id', 'category_id']

MODEL_DIR = "./models"
MODEL_NAME = "vgg19.npy"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
VGG_MEAN = [103.939, 116.779, 123.68]

RECONS_IMAGE_DIR = "recons_images"
SAVED_MODELS_DIR = "model_checkpoints"
RECONS_IMAGE_PATH = os.path.join(RESULT_DIR, RECONS_IMAGE_DIR)
SAVED_MODELS_PATH = os.path.join(RESULT_DIR, SAVED_MODELS_DIR)
