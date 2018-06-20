import time
import csv
from scipy import misc
import numpy as np
import h5py
from god_config import *
import tensorflow as tf
from vgg19_model import VGG19
import copy

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

def save_dnn_feature_map(image_ids, features, all_layers=True):
    if features == None:
        raise Exception("Input features is empty")
        return

    if all_layers:
        LAYER_TO_BE_SAVED = LAYER_TO_BE_SAVED_FULL
        RESULT_FILE = os.path.join(RESULT_DIR, IMAGE_FEATURES_FILE_NAME_FULL)
    else:
        LAYER_TO_BE_SAVED = LAYER_TO_BE_SAVED_LESS
        RESULT_FILE = os.path.join(RESULT_DIR, IMAGE_FEATURES_FILE_NAME_LESS)

    if len(LAYER_TO_BE_SAVED) != len(features):
        raise Exception("Length of names and features unmatches.")
    if len(image_ids) != (features[0].shape[0]):
        raise Exception("Number of images and features unmatches.")

    print ("Saving data...")
    start_time = time.time()
    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)
    for i in range(len(image_ids)):
        with h5py.File(RESULT_FILE, 'a') as f:
            grp = f.create_group(str(image_ids[i]))
            for j in range(len(LAYER_TO_BE_SAVED)):
                grp.create_dataset(LAYER_TO_BE_SAVED[j], data=features[j][i])
    print ("Data saved. takes %.2fs" %(time.time() - start_time))

def extract_dnn_features():
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

def get_dnn_features_by_imageid_and_layer(
        imageids, layers=None, all_layers=False):
    if all_layers:
        if not os.path.isfile(os.path.join( IMAGE_FEATURES_FILE_NAME_FULL)):
            raise Exception("Image feature file doesnt exist!")
        else:
            file_path = os.path.join(RESULT_DIR, IMAGE_FEATURES_FILE_NAME_FULL)
    else:
        if not os.path.isfile(os.path.join(RESULT_DIR, IMAGE_FEATURES_FILE_NAME_LESS)):
            raise Exception("Image feature file doesnt exist!")
        else:
            file_path = os.path.join(RESULT_DIR, IMAGE_FEATURES_FILE_NAME_LESS)

    if len(imageids) == 0:
        raise Exception("Image ids to be retrieved is none.")

    ret = {}
    with h5py.File(file_path, 'r') as f:
        for imageid in imageids:
            if imageid not in f:
                print ("{} is not in dataset.".format(imageid))
                continue
            else:
                if layers == None:
                    tmp_dict = {}
                    for key in list(f[imageid]):
                        tmp_dict[key] = np.array(f[imageid][key])
                    ret[imageid] = tmp_dict
                else:
                    tmp_dict = {}
                    for layer in layers:
                        if layer not in f[imageid]:
                            print ("Layer {} cannot be found in image {}".format(layer, imageid))
                        else:
                            tmp_dict[layer] = np.array(f[imageid][layer])
                    ret[imageid] = tmp_dict

    return ret

def recon_image_by_given_layer(reshaped_target, name):

    target = tf.placeholder(tf.float32, reshaped_target.shape)


    inputs = tf.Variable(tf.random_normal((1, 224, 224, 3)), name='recons_image')
    vgg19 = VGG19()
    vgg19.build(inputs, reshaped_target, name)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(vgg19.loss)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    feed_dict = {target: reshaped_target}

    num_of_epoches = 200000
    save_every = 10000
    # create folder to store the reconstructed images
    if not os.path.exists(RECONS_IMAGE_PATH):
        os.mkdir(RECONS_IMAGE_PATH)

    logs_path = os.path.join(LOGS_PATH, name)

    with tf.Session() as sess:
        sess.run(init)
        summary_w = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        for i in range(num_of_epoches):
            _, cost, summary = sess.run([optimizer, vgg19.loss, vgg19.merged_summary_ops], feed_dict=feed_dict)
            print ("Epoch %d/%d, cost: %.4f" % ((i+1), num_of_epoches, cost))
            if (i+1) % save_every == 0:
                start_time = time.time()
                print ("Start to save model...")
                saver.save(sess, SAVED_MODELS_PATH+'/model_ckpt/'+name, global_step=(i+1))
                print ("Model saved, takes %.3f" %(time.time()-start_time))
                summary_w.add_summary(summary, (i+1))
                im = (tf.get_default_graph().get_tensor_by_name('recons_image:0')).eval()[0,:,:,:]
                im = (im*255).astype(np.uint8)
                image_file_name = str(i+1)+'.jpg'
                if not os.path.exists(RECONS_IMAGE_PATH + '/' +name ):
                    os.mkdir(RECONS_IMAGE_PATH + '/' +name )
                img_path = RECONS_IMAGE_PATH + '/' + name + '/' + image_file_name
                misc.imsave(img_path, im)



