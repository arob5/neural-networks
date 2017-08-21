#
# tensorflow_intro.py
# Exploring the very basic functionality of tensorflow
# Last Modified: 8/20/2017
# Modified By: Andrew Roberts
#

import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)
result = tf.multiply(x1, x2)

with tf.Session() as sesh:
	output = sesh.run(result) 
	print(output)

