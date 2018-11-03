import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def net(input):
    x = input
    x = layers.Conv2D(16, 3, activation='relu')(x)
    x = layers.Conv2D(16, 3, activation='relu')(x)
    x = layers.Conv2D(10, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    return x


def main():
    print(tf.__version__)
    print(tf.keras.__version__)

    sample_shape = (1, 28, 28, 1)

    use_eager = True

    if use_eager:
        tf.enable_eager_execution()

        sample = tf.random_normal(sample_shape)
        result = net(sample)
        print(result)
        pass

    else:
        sample_ph = tf.placeholder(tf.float32, shape=sample_shape)
        graph_op = net(sample_ph)

        sample = np.random.randn(*sample_shape)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            result = sess.run(graph_op, feed_dict={sample_ph: sample})
            print(result)

        pass

    pass

if __name__ == "__main__":
    main()
