#
# run_speech_model.py
# Trains neural network on presidential speech data during election season
# Last Modified: 8/21/2017
# Modified By: Andrew Roberts
#

import tensorflow as tf
import numpy as np
import pickle

# Load Data
x_train, x_test, y_train, y_test = pickle.load(open("speech_data_preprocessed.pickle", "rb"))

# Parameters
learning_rate = .01
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_features = len(x_train[0])
n_classes = 2
batch_size = 100

x = tf.placeholder("float", [None, n_features]) 
y = tf.placeholder("float") 

def neural_net(data):
	
	# Weight matrix is (784, 500) -> 784 features and 500 nodes in first hidden layer
	hidden_layer_1 = {"weights": tf.Variable(tf.random_normal([n_features, n_nodes_hl1])),
			  "biases": tf.Variable(tf.random_normal([n_nodes_hl1]))}
	
	# Weight matrix is (500, 500) -> 500 nodes in hl1 and 500 nodes in hl2
	hidden_layer_2 = {"weights": tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
			  "biases": tf.Variable(tf.random_normal([n_nodes_hl2]))}
	
	# Weight matrix is (500, 500) -> 500 nodes in hl2 and 500 nodes in hl3
	hidden_layer_3 = {"weights": tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
			  "biases": tf.Variable(tf.random_normal([n_nodes_hl3]))}
	
	# Weight matrix is (500, 10) -> 500 nodes in hl3 and 10 nodes in one-hot encoded output layer
	output_layer = {"weights": tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
			  "biases": tf.Variable(tf.random_normal([n_classes]))}

	l1 = tf.add(tf.matmul(data, hidden_layer_1["weights"]), hidden_layer_1["biases"])
	l1 = tf.nn.relu(l1)
	
	l2 = tf.add(tf.matmul(l1, hidden_layer_2["weights"]), hidden_layer_2["biases"])
	l2 = tf.nn.relu(l2)
	
	l3 = tf.add(tf.matmul(l2, hidden_layer_3["weights"]), hidden_layer_3["biases"])
	l3 = tf.nn.relu(l3)

	output = tf.add(tf.matmul(l3, output_layer["weights"]), output_layer["biases"])
	
	return output

def train_neural_net(x):
	y_pred = neural_net(x)

	# Defining cost and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_pred, labels = y))
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

	n_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
	
		### Train ###
		for epoch in range(n_epochs):
			epoch_loss = 0

			for i in range(0, len(x_train), batch_size):
				start = i
				end = i + batch_size
				batch_xs = x_train[start:end]
				batch_ys = y_train[start:end]

				_, c = sess.run([optimizer, cost], feed_dict = {x: batch_xs, y: batch_ys})
				epoch_loss += c
			print("Epoch: {}; Loss: {}".format(epoch, epoch_loss))

		### Run optimized model ###
		correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, "float"))
		print("Accuracy:", accuracy.eval({x: x_test, y: y_test}))
		
train_neural_net(x)	
