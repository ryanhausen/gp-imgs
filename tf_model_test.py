import os

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import ImageGenerator as ig
from datahelper import DataHelper

from network import ResNet
import evaluate

home = os.getenv('HOME')

model_dir = os.path.join(home, 'Documents/synth_imgs/model/vals')
batch_size = 50
block_config = [2,4,16,8]

x = tf.placeholder(tf.float32, [batch_size, 84, 84, 4])
y = tf.placeholder(tf.float32, [batch_size, 5])

net = ResNet.build_graph(x, block_config, is_training=False)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

dh = DataHelper(batch_size=batch_size,
                    train_size=.80,
                    label_noise=False,
                    data_dir=os.path.join(home, 'Documents/galaxy-classification/data'),
                    transform_func=None)

with tf.Session() as sess:
    saver.restore(sess, os.path.join(model_dir, '-3889'))

    print('Synth-data')

    synth_xs, synth_ys = ig.generate_batch(batch_size)
    
    synth_max = np.max(synth_xs.reshape(batch_size, -1), axis=0)
    print(np.round(sess.run(net, feed_dict={x:synth_xs}), decimals=2))
    
    results = evaluate.evaluate(sess, net, x, y, synth_xs, synth_ys, 'eval_output.csv')
    print(results)
    
    print('Regular Data')
    
    batch_xs, batch_ys = dh.get_next_batch()
    reg_max = np.max(batch_xs.reshape(batch_size, -1), axis=0)
    print(np.round(sess.run(net, feed_dict={x:batch_xs}), decimals=2))
    
    results = evaluate.evaluate(sess, net, x, y, batch_xs, batch_ys, 'eval_output.csv')
    print(results)
    
f, ax = plt.subplots(4, 1)
for i in range(4):
    ax[i].imshow(synth_xs[0,:,:,i], cmap='gray')

for d in [('synth', synth_max), ('reg',reg_max)]:
    plt.figure()
    plt.title(d[0])
    plt.hist(d[1], bins=50)

plt.show()
