import os
# Verify that Tensorflow is working
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

# Print Version
print("Tensorflow version is " + str(tf.__version__))

# Verify session works
hello = tf.constant('Hello from Tensorflow')
sess = tf.Session()
print(sess.run(hello))
