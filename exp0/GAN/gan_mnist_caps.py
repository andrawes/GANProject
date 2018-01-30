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

# TODO: See the effect of changing routing iterations
from config import cfg
#from utils import load_data
#from capsNet import CapsNet
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
BATCH_SIZE = 50 # Batch size
batchsize = BATCH_SIZE
MASK_SIZE = 3
LEARNING_RATE = 1e-4 

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

# This should be identical to the squash function defined in capsLayer.py
def squash2(vector):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    epsilon = 1e-9
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)


# This method generates random noise as input to the generator
def caps_noise_generator(batch=50):
    vec_len = 16
    num_caps = 10
    # Instantiate mask
    mask = [None]*batch
    
    for i in range(0,batch):
        random_vector = np.random.normal(size=(1,vec_len))
        un_masked_caps = (int)(np.random.randint(size=(1,), low=0, high=num_caps))
        single_slice = np.lib.pad(random_vector, pad_width=((un_masked_caps, num_caps-1-un_masked_caps),(0,0)), mode='constant')
        mask[i] = single_slice
    # Transform into numpy array
    mask = np.array(mask)
    # This is a [bs, 10, 16] tensor
    noise = tf.convert_to_tensor(mask, np.float32)
    # reshape to [batch, num_caps=10, vec_len = 16, 1]
    noise = tf.reshape(noise, shape=[batch, num_caps, vec_len, 1])
    # Squash noise
    noise = squash2(noise) # This of shape [batch, num_caps=10, vec_len = 16, 1] 

    return noise





def Generator(n_samples, noise=None, reuse=False):

    with tf.variable_scope('CapsGener', reuse=reuse):
         
        # TODO: Play around with different ways of generating noise, either masked, or unmasked.
        # Masked noise generation
        if noise is None:
            noise = caps_noise_generator(batch=n_samples)  
            # This is of shape [batch, num_caps=10, vec_len = 16, 1] 
        
        # Inverted digitcaps to primary caps fully connect layer with dynamic routing
        dedigitCaps = CapsLayer(num_outputs=1152, vec_len=8, with_routing=True, layer_type='FC')
        output_caps2 = dedigitCaps(noise, batchsize=n_samples)

        print("Output_caps2 dimensions should be : (bs, 1152,8,1), and actually are: ")
        print(output_caps2.get_shape())

        # TODO: Try adding convolutional capsule layers to the network.
        
        reshape1 = tf.reshape(output_caps2, (n_samples,32,6,6,8))
        # Not sure if the two following lines are actually useful
        transposed = tf.transpose(reshape1, [0, 1, 4, 2, 3])
        reshaped_for_deconv = tf.reshape(transposed, (n_samples,256,6,6))

        # TODO: Determine appropriate shapes
        print("Reshaped_for_deconv caps2 dimensions should be : (b_s, 256, 6,6), and actually are: ")
        print(reshaped_for_deconv.get_shape())
    
        # Deconvolution 1
        deconv1 = lib.ops.deconv2d.Deconv2D('CapsDeconv1', 256, 256, MASK_SIZE, reshaped_for_deconv)

        paddings = tf.constant([[0, 0,], [0, 0], [1, 1], [1, 1]])
        padded = tf.pad(deconv1, paddings, "SYMMETRIC")

        # Make sure if we want to have this ReLU or not
        relu_1 = tf.nn.relu(padded)

        # TODO: Get shape of deconv1 output

        # Deconvolution 2
        # TODO: Change the input accordingly if we want to have the ReLU or not
        deconv2 = lib.ops.deconv2d.Deconv2D('CapsDeconv2', 256, 1, MASK_SIZE, relu_1)

        print("The shape of deconv2 should be (b_s,1,28,28) :")
        print((deconv2.get_shape()))

        # Make sure if we want to have this ReLU or not (This one probably not if there is the sigmoid)
        # relu_2 = tf.nn.relu(deconv2)

        # TODO: Check if we want the sigmoid output or not
        sigmoid_out = tf.nn.sigmoid(deconv2)
    
        print("Shape at the output:")
        print(sigmoid_out.get_shape())

        # OUTPUT_DIM is the number of pixels in MNIST aka 784
        return tf.reshape(sigmoid_out, [-1, OUTPUT_DIM])




def Discriminator(inputs):
    output = tf.reshape(inputs, [-1, 1, 28, 28])

    output = lib.ops.conv2d.Conv2D('Discriminator.1',1,DIM,5,output,stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 1, output)

    return tf.reshape(output, [-1])


    

real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
fake_data = Generator(BATCH_SIZE, reuse=False)

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

# gen_params = lib.params_with_name('Generator')
 
trainable_vars = tf.trainable_variables()
gen_params = [var for var in trainable_vars if var.name.startswith("CapsGener")]

disc_params = lib.params_with_name('Discriminator')

if MODE == 'wgan':
    pass

elif MODE == 'wgan-gp':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1], 
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty

    # TODO : Try Playing with Adamopt parameters
    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=LEARNING_RATE, 
        beta1=0.5,
        beta2=0.9
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5, 
        beta2=0.9
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

elif MODE == 'dcgan':
    pass

# For saving samples
# TODO: Try to change how noise is masked

fixed_noise = caps_noise_generator(batch=128)
#fixed_noise = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
fixed_noise_samples = Generator(128, noise=fixed_noise, reuse=True)
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
            _disc_cost, _ , _gen_cost= session.run(
                [disc_cost, disc_train_op, gen_cost],
                feed_dict={real_data: _data}
            )
            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)

        lib.plot.plot('gen disc cost', _gen_cost)
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
