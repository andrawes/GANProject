import os, sys
#os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import tensorflow as tf

from config import cfg
from utils import load_data
from capsNet import CapsNet
from capsLayer import CapsLayer


import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist
import tflib.plot

MODE = 'wgan-gp' # dcgan, wgan, or wgan-gp
DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # MAKE SURE BATCH SIZE IS CHANGED HERE AND IN CONFIG FILE

CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many generator iterations to train for 
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)

lib.print_model_settings(locals().copy())









def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)

def Generator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM, noise)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = output[:,:,:7,:7]

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 1, 5, output)
    output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])






def Discriminator(inputs, reuse=False, batchsize=BATCH_SIZE):
    with tf.variable_scope('CapsDiscrim', reuse=reuse):

        output = tf.reshape(inputs, [-1, 1, 28, 28])
        # The following line flips the dimensions so that the CapsNet architecture
        # can be kept as is. 

        # This command re-orders the dimensions
        output = tf.transpose(output, [0, 2, 3, 1])
        # TODO: make sure the shape is correct

        #with tf.variable_scope('Conv1_layer'):
            # Conv1, [batch_size, 20, 20, 256]
        output_conv1 = tf.contrib.layers.conv2d(output, num_outputs=256,
                                         kernel_size=9, stride=1,
                                         #activation_fn=None,				# Added this line to remove the ReLU activation
                                         activation_fn=tf.nn.relu,
                                         padding='VALID')

        #output_LeakyReLU = LeakyReLU(output_conv1)

        # Primary Capsules layer, return [batch_size, 1152, 8, 1]

        primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV')
       
        #output_caps1 = primaryCaps(output_LeakyReLU, kernel_size=9, stride=2, batchsize=batchsize)
        output_caps1 = primaryCaps(output_conv1, kernel_size=9, stride=2, batchsize=batchsize)    

        # DigitCaps layer, return [batch_size, 10, 16, 1]
        #with tf.variable_scope('DigitCaps_layer'):
        digitCaps = CapsLayer(num_outputs=1, vec_len=16, with_routing=True, layer_type='FC')
        output_caps2 = digitCaps(output_caps1, batchsize=batchsize)

        # The output at this stage is of dimensions [batch_size, 16]
        output_caps2 = tf.squeeze(output_caps2, axis=1)
        output_caps2 = tf.squeeze(output_caps2, axis=2)
       
        #print(output_caps2.get_shape())
        assert output_caps2.get_shape() == [batchsize, 16]

    	# TODO: Try also removing the LeakyReLU from the CapsLayer file
        # TODO: Try also with 10 digitcaps outputs + thresholding (instead of just 1 output)
        # TODO: Adding batch normalization in capsules (See CapsLayer.py). 
        # TODO: Try Changing the critic iteration count.

        output_v_length = tf.sqrt(tf.reduce_sum(tf.square(output_caps2),axis=1, keep_dims=True) + 1e-9)

        ## No need to take softmax anymore, because output_caps2 output is in [0,1] due to squash function.   
        #softmax_v = tf.nn.softmax(v_length, dim=1)

        return tf.reshape(output_v_length, [-1])


############################## Beginning of the main execution ############################
###########################################################################################

 
real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
fake_data = Generator(BATCH_SIZE)

print("DEBUG: Real and fake data shapes")
print(real_data.get_shape())
print(fake_data.get_shape())

print("FLAGPOINT: BEFORE DISC REAL")
disc_real = Discriminator(real_data, reuse=False)
print("FLAGPOINT: AFTER DISC REAL")
disc_fake = Discriminator(fake_data, reuse=True)
print("FLAGPOINT: AFTER DISC FAKE")

gen_params = lib.params_with_name('Generator')

# Obtain parameters differently for disciminator (we used variable scope previously)
# disc_params = lib.params_with_name('Discriminator')
trainable_vars = tf.trainable_variables()
disc_params = [var for var in trainable_vars if var.name.startswith("CapsDiscrim")]


if MODE == 'wgan':
    pass

elif MODE == 'wgan-gp':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    # TODO: Otherwise, do we have to normalize the output of the discriminator 

    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1], 
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)

    print(interpolates.get_shape())
   
    gradients = tf.gradients(Discriminator(interpolates, reuse=True, batchsize=50), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5,
        beta2=0.9
    ).minimize(gen_cost, var_list=gen_params)


    # TODO: Adam optimizer params
    disc_train_op = tf.train.AdamOptimizer(learning_rate=5e-6, beta1=0.5,beta2=0.9).minimize(disc_cost, var_list=disc_params)


    clip_disc_weights = None

elif MODE == 'dcgan':
    pass

# For saving samples
fixed_noise = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
fixed_noise_samples = Generator(128, noise=fixed_noise)
def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples)
    lib.save_images.save_images(
        samples.reshape((128, 28, 28)), 
        'samples_{}.png'.format(frame)
    )

# Dataset iterator
train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)
def inf_train_gen():
    while True:
        for images,targets in train_gen():
            yield images

# Train loop
with tf.Session() as session:

    session.run(tf.initialize_all_variables())

    gen = inf_train_gen()

    for iteration in xrange(ITERS):
        start_time = time.time()

        if iteration > 0:
            _ = session.run(gen_train_op)

        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in xrange(disc_iters):

            _data = gen.next()
            # 
            _disc_cost, _ = session.run([disc_cost, disc_train_op],feed_dict={real_data: _data})

            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)

        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            for images,_ in dev_gen():
                _dev_disc_cost = session.run(
                    disc_cost, 
                    feed_dict={real_data: images}
                )
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))

            generate_image(iteration, _data)

        # Write logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush()

        lib.plot.tick()
