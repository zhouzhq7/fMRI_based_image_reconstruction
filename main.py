from utils import read_images
import time
from vgg19_model import VGG19
import tensorflow as tf


def main():
    data = read_images()
    data_shape = data["rescaled_images"].shape
    batch_size = 10
    num_of_batch = int(data_shape[0]/batch_size)+1
    
    with tf.Session() as sess:
        inputs = tf.placeholder("float", (None, 224, 224, 3))

        vgg19 = VGG19()

        for i in range(num_of_batch):
            if i == num_of_batch - 1:
                rgb = (data["rescaled_images"][i*batch_size:, :, :, :]).astype("float32")
            else:
                rgb = (data["rescaled_images"][i*batch_size:(i+1)*batch_size,:, :, :]).astype("float32")
            with tf.name_scope("vgg_content"):
                vgg19.build(rgb)
            feed_dict = {inputs: rgb}
            start_time = time.time()

            print ("Start to evalutate batch {}".format(i))
            layers = sess.run([vgg19.conv1_2, vgg19.conv2_2,
                           vgg19.conv3_4, vgg19.conv4_4,
                           vgg19.conv5_4, vgg19.fc6,
                           vgg19.fc7, vgg19.fc8], feed_dict=feed_dict)
            for layer in layers:
                print (layer.shape)
            print ("Time spent : %.5ss" %(time.time()-start_time))



if __name__=="__main__":
    main()
