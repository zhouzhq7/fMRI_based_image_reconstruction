from utils import *
import time
from vgg19_model import VGG19
import tensorflow as tf
import numpy as np
from scipy import misc

def main():
    target_feature_map = get_dnn_features_by_imageid_and_layer(['11978233.018962'], ['conv5_4'])
    reshaped_target = (target_feature_map['11978233.018962']['conv5_4']).reshape((1, 14, 14, 512))
    target = tf.placeholder(tf.float32, reshaped_target.shape)

    inputs = tf.Variable(tf.random_normal((1, 224, 224, 3)), name='recons_image')

    vgg19 = VGG19()

    vgg19.build(inputs, reshaped_target)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(vgg19.loss)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    feed_dict = {target: reshaped_target}

    num_of_epoches = 20000
    save_every = 100
    # create folder to store the reconstructed images
    if not os.path.exists(RECONS_IMAGE_PATH):
        os.mkdir(RECONS_IMAGE_PATH)

    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_of_epoches):
            _, cost = sess.run([optimizer, vgg19.loss], feed_dict=feed_dict)
            print ("Epoch %d/%d, cost: %.4f" % ((i+1), num_of_epoches, cost))
            if (i+1) % save_every == 0:
                start_time = time.time()
                print ("Start to save model...")
                saver.save(sess, SAVED_MODELS_PATH+'/model_ckpt', global_step=(i+1))
                print ("Model saved, takes %.3f" %(time.time()-start_time))
                im = (tf.get_default_graph().get_tensor_by_name('recons_image:0')).eval()[0,:,:,:]
                im = (im*255).astype(np.uint8)
                image_file_name = str(i+1)+'.jpg'
                img_path = RECONS_IMAGE_PATH + '/' + image_file_name
                misc.imsave(img_path, im)

if __name__=="__main__":
    main()
    print ("Good good study, day day up!")
