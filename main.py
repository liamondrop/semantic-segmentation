import os.path
import tensorflow as tf
import numpy as np
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

DATA_DIR = './data'
RUNS_DIR = './runs'
IMG_SHAPE = (160, 576)
NUM_CLASSES = 2
NUM_EPOCHS = 30
BATCH_SIZE = 1
KEEP_PROB = 0.5
LEARNING_RATE = 0.0001
LINE_LEN = 25


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # load the model and weights
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    def conv_layer(inputs, name):
        return tf.layers.conv2d(inputs, num_classes, kernel_size=(1,1), strides=(1,1), name=name)

    def upsample_layer(inputs, name, k, s):
        return tf.layers.conv2d_transpose(inputs, num_classes, kernel_size=(k,k), strides=(s,s), padding='same', name=name)

    # 1x1 convolution of vgg layers
    conv_layer3 = conv_layer(vgg_layer3_out, 'conv_layer3')
    conv_layer4 = conv_layer(vgg_layer4_out, 'conv_layer4')
    conv_layer7 = conv_layer(vgg_layer7_out, 'conv_layer7')

    # Add decoder layers to the model with upsampling and skip connections
    decoder_layer1 = upsample_layer(conv_layer7, 'decoder_layer1', k=4, s=2)
    decoder_layer2 = tf.add(decoder_layer1, conv_layer4, name='decoder_layer2')
    decoder_layer3 = upsample_layer(decoder_layer2, 'decoder_layer3', k=4, s=2)
    decoder_layer4 = tf.add(decoder_layer3, conv_layer3, name='decoder_layer4')
    decoder_output = upsample_layer(decoder_layer4, 'decoder_output', k=16, s=8)
    return decoder_output


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # Reshape 4D outputs to 2D, in which each row represents a pixel and each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    # Define cross entropy loss function
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)

    # Find the weights and parameters to yield correct pixel labels
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss


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
    for epoch in range(epochs):
        print('EPOCH: {}'.format(epoch))
        print('=' * LINE_LEN)

        losses_in_epoch = []
        for i, batch in enumerate(get_batches_fn(batch_size)):
            images, labels = batch
            feed_dict = {
                input_image: images,
                correct_label: labels,
                keep_prob: KEEP_PROB,
                learning_rate: LEARNING_RATE}

            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
            losses_in_epoch.append(loss)
            print('Batch {:02d} Loss: {:.5f}'.format(i, loss))
        
        mean_epoch_loss = np.mean(losses_in_epoch)
        print('-' * LINE_LEN)
        print('Mean Loss for Epoch {}: {:.5f}'.format(epoch, mean_epoch_loss))
        print('-' * LINE_LEN)


def run():
    # Paths to vgg model and training data
    vgg_path = os.path.join(DATA_DIR, 'vgg')
    training_path = os.path.join(DATA_DIR, 'data_road/training')

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(DATA_DIR)

    # Create function to get batches
    get_batches_fn = helper.gen_batch_function(training_path, IMG_SHAPE)

    with tf.Session() as sess:
        # Placeholders
        num_rows, num_cols = IMG_SHAPE
        correct_label = tf.placeholder(tf.int32, [None, num_rows, num_cols, NUM_CLASSES], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # Load the image input, keep probability and vgg layers from the vgg architecture
        image_input, keep_prob, vgg_layer3, vgg_layer4, vgg_layer7 = load_vgg(sess, vgg_path)

        # Add a decoder layer to the vgg model
        decoder_output = layers(vgg_layer3, vgg_layer4, vgg_layer7, NUM_CLASSES)

        # Build the loss and optimizer operations
        logits, train_op, cross_entropy_loss = optimize(decoder_output, correct_label, learning_rate, NUM_CLASSES)

        # Initialize the TF Variables
        sess.run(tf.global_variables_initializer())

        # Train the network
        train_nn(sess, NUM_EPOCHS, BATCH_SIZE, get_batches_fn,
                 train_op, cross_entropy_loss, image_input,
                 correct_label, keep_prob, learning_rate)

        # Save inference samples on test images
        helper.save_inference_samples(RUNS_DIR, DATA_DIR, sess, IMG_SHAPE, logits, keep_prob, image_input)

def run_tests():
    tests.test_load_vgg(load_vgg, tf)
    tests.test_layers(layers)
    tests.test_optimize(optimize)
    tests.test_train_nn(train_nn)
    tests.test_for_kitti_dataset(DATA_DIR)
    print('All Tests Passed')
    print('=' * LINE_LEN)


if __name__ == '__main__':
    run_tests()
    run()
