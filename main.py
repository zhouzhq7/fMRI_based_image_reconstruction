from utils import *
import time
from vgg19_model import VGG19
import tensorflow as tf
import copy
import numpy as np

def main():
    data = read_images()
    data_shape = data["rescaled_images"].shape
    batch_size = 15
    num_of_batch = int(data_shape[0]/batch_size)+1

    save_every = 4

    inputs = tf.placeholder("float", (None, 224, 224, 3))

    vgg19 = VGG19()

    with tf.name_scope("vgg_content"):
        vgg19.build(inputs)
    layers_so_far = []
    images_id_so_far = []
    for i in range(num_of_batch):
        if i == num_of_batch - 1:
            rgb = (data["rescaled_images"][i*batch_size:, :, :, :]).astype("float32")
            images_id_so_far += data["image_ids"][0][i*batch_size:]
        else:
            rgb = (data["rescaled_images"][i*batch_size:(i+1)*batch_size,:, :, :]).astype("float32")
            images_id_so_far += data["image_ids"][0][i*batch_size:(i+1)*batch_size]

        feed_dict = {inputs: rgb}
        start_time = time.time()

        with tf.Session() as sess:
            print ("Start to evaluate batch {}/{}".format(i+1, num_of_batch))
            layers = sess.run([vgg19.conv1_2, vgg19.conv2_2,
                               vgg19.conv3_4,
                               vgg19.conv4_4,
                               vgg19.conv5_4,
                               vgg19.fc6, vgg19.fc7, vgg19.fc8],
                               feed_dict=feed_dict)

            print ("Time spent : %.5ss" %(time.time()-start_time))

            if len(layers_so_far) == 0:
                layers_so_far = copy.deepcopy(layers)
            else:
                if len(layers_so_far) != len(layers):
                    raise Exception('Configurations doesn\'t match previous')
                else:
                    for j in range(len(layers_so_far)):
                        layers_so_far[j] = np.concatenate((layers_so_far[j], layers[j]) , axis=0)
        if (i+1) % save_every == 0 or i == num_of_batch-1:
            save_dnn_feature_map(features=layers_so_far, image_ids=images_id_so_far, all_layers=False)
            layers_so_far = []
            images_id_so_far = []



if __name__=="__main__":
    main()
