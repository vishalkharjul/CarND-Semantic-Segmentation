import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from moviepy.editor import VideoFileClip
import time
import scipy.misc
import numpy as np 

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess,[vgg_tag],vgg_path)
    
    graph = tf.get_default_graph()
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob_tensor = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out_tensor = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out_tensor = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out_tensor = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    
    return input_tensor, keep_prob_tensor, layer3_out_tensor, layer4_out_tensor, layer7_out_tensor
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    Layer7_conv_1x1_op = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1),padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),            
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    print(Layer7_conv_1x1_op.get_shape())
    layer4_conv_trans_1 = tf.layers.conv2d_transpose(Layer7_conv_1x1_op, num_classes, 4, strides=2,padding='same',
                                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),  
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    print(layer4_conv_trans_1.get_shape())
    
    layer4_conv_trans_2 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1,1),padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),             
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    print(layer4_conv_trans_2.get_shape())
    
    
    conv_skip_1 = tf.add(layer4_conv_trans_1, layer4_conv_trans_2)
    
    
    print(conv_skip_1.get_shape())
    
    layer3_conv_trans_1 = tf.layers.conv2d_transpose(conv_skip_1, num_classes, 4, strides=2,padding='same',
                                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),        
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    print(layer3_conv_trans_1.get_shape())
    
    layer3_conv_trans_2 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1,1),padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),             
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    print(layer3_conv_trans_2.get_shape())
    
    conv_skip_2 = tf.add(layer3_conv_trans_1, layer3_conv_trans_2) 
    
    print(conv_skip_2.get_shape())
    
    output = tf.layers.conv2d_transpose(conv_skip_2, num_classes, 16, strides=8,padding='same',
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),  
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1,num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    
    return  logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    
    sess.run(tf.global_variables_initializer())
    tic = time.time()
    
    
    print("Training...")
    print()
    for i in range(epochs):
        print("EPOCH {} ...".format(i+1))
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], 
                               feed_dict={input_image: image, correct_label: label,keep_prob: 0.5, learning_rate: 0.0009})
            print("Loss: = {:.3f}".format(loss))
        print()
    toc = time.time()
    print('computed in %fs' % (toc - tic))
    
    
             
        
        
tests.test_train_nn(train_nn)


def process_image(input_image):
    image_shape = (160, 576)
    image = scipy.misc.imresize(input_image, image_shape)
    
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('FCNVGG16.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        print("Model restored.")
        graph = tf.get_default_graph()
        image_input = graph.get_operation_by_name("image_input").outputs[0]
        logits=graph.get_operation_by_name("logits").outputs[0]
        keep_prob_tensor = graph.get_operation_by_name("keep_prob").outputs[0]
        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob_tensor: 1.0, image_input: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)
    return street_im
           


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        
        epochs = 20
        batch_size = 8
        
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        
        # TODO: Build NN using load_vgg, layers, and optimize function
        
        input_tensor, keep_prob_tensor, layer3_out_tensor, layer4_out_tensor, layer7_out_tensor = load_vgg(sess, vgg_path)
        output = layers(layer3_out_tensor, layer4_out_tensor, layer7_out_tensor,num_classes)
        logits, train_op, cross_entropy_loss = optimize(output, correct_label, learning_rate, num_classes)
        
        # TODO: Train NN using the train_nn function
        saver = tf.train.Saver(None)
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_tensor,
             correct_label, keep_prob_tensor, learning_rate)
        saver.save(sess, './FCNVGG16')
        print("Model saved")
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob_tensor, input_tensor)
        

        # OPTIONAL: Apply the trained model to a video
        

        

        



if __name__ == '__main__':
    run()
    #Code to apply it on Video
    #white_output = 'white.mp4'
    #clip1 = VideoFileClip("/home/vishal/SDCND/CarND-Semantic-Segmentation/solidWhiteRight.mp4")
    #processed_clip = clip1.fl_image(process_image)
    #processed_clip.write_videofile(white_output, audio=False)
