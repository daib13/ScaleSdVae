import tensorflow as tf 
from tensorflow.contrib import layers
from nn import res_block


x = tf.placeholder(tf.float32, [10, 32, 32, 3])
with tf.variable_scope('outer_loop1'):
    y1 = res_block('weight', x)

with tf.variable_scope('outer_loop2'):
    y2 = res_block('weight', x)

print('...')
add a line