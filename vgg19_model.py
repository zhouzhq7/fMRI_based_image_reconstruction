import numpy as np
import time
import tensorflow as tf
from god_config import *

class VGG19:
    def __init__(self, model_path=MODEL_PATH):
        print ("Load vgg19 model from {}".format(model_path))
        start_time = time.time()

        self.params_dict = np.load(model_path, encoding='latin1').item()

        print ("Model loaded, takes %ds." % (time.time()-start_time))

    def build(self, rgb, target, name, train=False, use_prior=False, use_all_layers=False):
        start_time = time.time()
        print ("Start to build model....")
        rgb_rescaled = rgb * 255.0

        r, g, b = tf.split(value=rgb_rescaled, num_or_size_splits=3, axis=3)

        # transform rgb into bgr
        assert r.get_shape().as_list()[1:] == [224, 224, 1]
        assert b.get_shape().as_list()[1:] == [224, 224, 1]
        assert g.get_shape().as_list()[1:] == [224, 224, 1]

        #print ("shape of single channel is {}".format(b.get_shape()))

        bgr = tf.concat(
            values=[
                b - VGG_MEAN[0],
                g - VGG_MEAN[1],
                r - VGG_MEAN[2]
            ],
            axis=3
        )

        assert bgr.get_shape().as_list()[1:] == [224,224,3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        if train:
            self.pool1 = self.max_pool(self.conv1_2, "pool1")
        else:
            self.pool1 = self.avg_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        if train:
            self.pool2 = self.max_pool(self.conv2_2, 'pool2')
        else:
            self.pool2 = self.avg_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
        if train:
            self.pool3 = self.max_pool(self.conv3_4, 'pool3')
        else:
            self.pool3 = self.avg_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
        if train:
            self.pool4 = self.max_pool(self.conv4_4, 'pool4')
        else:
            self.pool4 = self.avg_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
        if train:
            self.pool5 = self.max_pool(self.conv5_4, 'pool5')
        else:
            self.pool5 = self.avg_pool(self.conv5_4, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, "fc8")

        self.loss = tf.convert_to_tensor(0.0, tf.float32)
        if use_all_layers:
             print ("using all layers")
             for i in range(len(LAYER_TO_BE_SAVED_LESS)):
                 target_pred = self.get_layer_by_name(LAYER_TO_BE_SAVED_LESS[i])
                 self.loss += tf.divide(tf.reduce_mean(tf.losses.mean_squared_error(target[i], target_pred)),
                                        tf.reduce_sum(target[i]))
        else:
            target_pred = self.get_layer_by_name(name)
            self.loss += tf.reduce_mean(tf.losses.mean_squared_error(target, target_pred))
        if use_prior:
            im = tf.get_default_graph().get_tensor_by_name('recons_image:0')
            tv_image = tf.image.total_variation(im)
            self.loss += tv_image

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.params_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    def get_layer_by_name(self, name):
        if name == 'conv1_2':
            return self.conv1_2
        elif name == 'conv2_2':
            return self.conv2_2
        elif name == 'conv3_4':
            return self.conv3_4
        elif name == 'conv4_4':
            return self.conv4_4
        elif name == 'conv5_4':
            return self.conv5_4
        elif name == 'fc6':
            return self.fc6
        elif name == 'fc7':
            return self.fc7
        elif name == 'fc8':
            return self.fc8
        else:
            raise Exception("{} is not supported in this version.".format(name))

    def conv_layer(self, prev, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(prev, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu


    def max_pool(self, prev, name):
        return tf.nn.max_pool(prev, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)

    def avg_pool(self, prev, name):
        return tf.nn.avg_pool(prev, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


    def fc_layer(self, prev, name):
        with tf.variable_scope(name):
            shape = prev.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim = dim * d

            x = tf.reshape(prev, [-1, dim])

            weights = self.get_fc_weight(name)
            bias = self.get_bias(name)

            fc = tf.nn.bias_add(tf.matmul(x, weights), bias)

            return fc


    def get_conv_filter(self, name):
        return tf.constant(self.params_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.params_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.params_dict[name][0], name="weights")

