# Introduction to Artificial Intelligence
# Introductory example about the use of TensorFlow
# By Juan Carlos Rojas

import tensorflow as tf

# Define variables
w = tf.constant(3)

# Define computation graph
x = w+2
y = x+5
z = x*3

# Create and initialize the session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Run the session
result = sess.run([y, z])
sess.close()

print(result)
