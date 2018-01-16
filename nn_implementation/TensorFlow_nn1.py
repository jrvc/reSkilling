# -*- coding: utf-8 -*-
"""
TensorFlow 
    implementation of the neural network in their tutorial - written numbers classification 
    The MNIST (Modified National Institute of Standards and Technology) database
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random 

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

''' V I S U A L I Z A T I O N '''
# view some of the trainig data:
for i in range(25):
    pic = round(random.uniform(0,len(mnist.train.images)))
    pl = plt.subplot(5,5,i+1)
    pl = plt.imshow(mnist.train.images[pic].reshape(28,28))
    pl.axes.get_xaxis().set_ticks([])
    pl.axes.get_yaxis().set_ticks([])

''' M O D E L   D E F I N I T I O N'''
x = tf.placeholder(tf.float32, [None, 784]) # None means the dimension can be of any length
# tf.placeholder defines a value that shall be given when tf shall run a computation

# weights ad biases
W = tf.Variable(tf.zeros([784, 10])) #tf.Variable defines a global variable
b = tf.Variable(tf.zeros([10]))

# definition of the model
y = tf.nn.softmax(tf.matmul(x, W) + b) # application of softmax to 1 layer. matmul == matrix multiplication


''' T R A I N I N G '''
# y_ is a placeholder for the observed data
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# tf.reduce_sum adds the elements in the second dimension of y, due to the reduction_indices=[1] parameter
'''NUMERICALLY STABLE VERSION OF CROSS ENTROPY FUNCTION:
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
'''
# minimize cross_entropy using the gradient descent algorithm
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

''' I N T E R A C T I V E    S E S S I O N'''
sess = tf.InteractiveSession()

# operation to initialize the variables we created
tf.global_variables_initializer().run()

# TRAIN 1000 STEPS
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) 

''' E V A L U A T I O N '''
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

sess.close()

'''--------------------------------------------------------'''
''' ------------------- P A R T   2 -----------------------'''

# Instead of creating W and b for each layer, create two handy functions:
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1) #Outputs random values from a truncated normal.
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# CONVOLUTION. uses a stride of one and are zero padded so that the output is the same size as the input
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') #Computes a 2-D convolution given 4-D `input` and `filter` tensors

# POOLING. plain old max pooling over 2x2 blocks 
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  
''' 1st  L A Y E R'''
# CONVOLUTION FOLLOWED BY MAX_POOLING
W_conv1 = weight_variable([5, 5, 1, 32]) # 5x5 patch, 1 input channel and 32 output channels
b_conv1 = bias_variable([32])  # bias vector with 32 output channels

x_image = tf.reshape(x, [-1, 28, 28, 1]) # -1 is used to infer the shape, 28x28 image size and 1 color channel
# If one component of `shape` is the special value -1, 
# the size of that dimension is computed so that the total size remains constant.

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # apply RelU to b_conv1 plus the convolution of x_image with the W_conv1 filter
h_pool1 = max_pool_2x2(h_conv1) # max_pool_2x2 method will reduce the image size to 14x14

''' 2nd  L A Y E R'''
W_conv2 = weight_variable([5, 5, 32, 64]) # The second layer will have 64 features for each 5x5 patch (32 inout features, coming form layer 1)
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) # this second max_pool_2x2 will reduce the image size to 7x7

''' D E N S E L Y   C O N N E C T E D    L A Y E R '''
#  fully-connected layer with 1024 neurons to allow processing on the entire image.
W_fc1 = weight_variable([7 * 7 * 64, 1024]) # h_pool2 is of size 7x7 and has 64 outputs
b_fc1 = bias_variable([1024]) # notice that 32*32 = 1024

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # flatten h_pool2
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) # multiply by a weight matrix, add a bias, and apply a ReLU


''' D R O P O U T '''
# to avoid overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) # with probability `keep_prob`, outputs the input element scaled up by `1 / keep_prob`, otherwise outputs `0`.



''' R E A D O U T   L A Y E R ''' 
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
 
''' T R A I N   &   E V A L '''
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # ADAM instead of GradDescent optimizing
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(10000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0: # this is for the logging
      train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))




