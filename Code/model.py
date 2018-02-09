import helper


#function to seperate sppech
def seperate_speech(audios):
	"""
	retuns index for speech type
	"""
	i = 0
	index = []
	for audio in audios:
		zcr = helper.zcr(audio[:1000])[0]
		if len(filter(lambda x: x >0.35 ,zcr)) > len(zcr)*.05:
			index.append(i)
		i=i+1
	return index

#function to seperate music from noise
def seperate_music1(tr_features, tr_labels,ts_features,ts_labels):
	"""
	print cross validation score for mlp
	"""
	from sklearn.neural_network import MLPClassifier
	clf = MLPClassifier()
	clf.fit(tr_features,tr_labels)
	a = clf.predict(ts_features)
	from sklearn.model_selection import cross_val_score
	print cross_val_score(clf,tr_features,tr_labels)


#function to seperate music from noise
def seperate_music3(tr_features, tr_labels,ts_features,ts_labels):
	"""
	print cross validation score for rnn model	
	"""
	import tensorflow as tf 
	import numpy as np

	tf.reset_default_graph()

	learning_rate = 0.01
	training_iters = 1000
	batch_size = 5
	epoch = 2

	# Network Parameters
	n_input = len(tr_features[0])
	n_hidden = 900
	n_classes = 1

	x = tf.placeholder("float", [None, n_input,1])
	y = tf.placeholder("float", [None, n_classes])

	weight = tf.Variable(tf.random_normal([n_hidden, n_classes]))
	bias = tf.Variable(tf.random_normal([n_classes]))




	def RNN(x, weight, bias):
		cell = tf.contrib.rnn.LSTMCell(n_hidden,state_is_tuple = True)
		outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
		output = tf.transpose(outputs, [1, 0, 2])
		last = tf.gather(output, int(output.get_shape()[0]) - 1)
		return tf.nn.softmax(tf.matmul(last, weight) + bias)
		

	prediction = RNN(x, weight, bias)

	# Define loss and optimizer
	loss_f = -tf.reduce_sum(y * tf.log(prediction))
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss_f)

	# Evaluate model
	correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	l_a = len(tr_features)
	# Initializing the variables
	init = tf.global_variables_initializer()
	import time
	a = time.time()
	with tf.Session() as session:
		session.run(init)
		
		for itr in range(training_iters):  
			for i in range(1,l_a/batch_size):  
				batch_x = tr_features[i*batch_size:i*batch_size+batch_size]
				batch_y = tr_labels[i*batch_size:i*batch_size+batch_size]

				_, c = session.run([optimizer, loss_f],feed_dict={x: batch_x, y : batch_y})
					
				if epoch % display_step == 0:
					# Calculate batch accuracy
					acc = session.run(accuracy, feed_dict={x: batch_x, y: batch_y})
					# Calculate batch loss
					loss = session.run(loss_f, feed_dict={x: batch_x, y: batch_y})
					print "Iter " + str(epoch) + ", Minibatch Loss= " + \
						  "{:.6f}".format(loss) + ", Training Accuracy= " + \
						  "{:.5f}".format(acc)
		
		print('Test accuracy: ',round(session.run(accuracy, feed_dict={x: ts_features, y: ts_labels}) , 3))
	print time.time() - accuracy 

#function to seperate music from noise
def seperate_music2(tr_features, tr_labels,ts_features,ts_labels):
	"""
	print cross validation score for rnn model	
	"""
	import tensorflow as tf 
	import numpy as np

	tf.reset_default_graph()

	learning_rate = 0.01
	training_iters = 1000
	batch_size = 5
	epoch = 2

	# Network Parameters
	n_input = len(tr_features[0])/100
	n_steps = 100
	n_hidden = 900
	n_classes = 1

	x = tf.placeholder("float", [None, n_input,100])
	y = tf.placeholder("float", [None, n_classes])

	weight = tf.Variable(tf.random_normal([n_hidden, n_classes]))
	bias = tf.Variable(tf.random_normal([n_classes]))




	def RNN(x, weight, bias):
		# print x,'x tensor'
		x = tf.transpose(x, [1, 0, 2])
		x = tf.reshape(x, [-1, n_input])
		x = tf.split(x, n_steps, 0)
		cell = tf.contrib.rnn.LSTMCell(n_hidden,state_is_tuple = True)
		# cell = tf.contrib.rnn.MultiRNNCell([cell, cell], state_is_tuple = True)
		outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
		output = tf.transpose(outputs, [1, 0, 2])
		last = tf.gather(output, int(output.get_shape()[0]) - 1)
		# last = outputs[-1]
		return tf.nn.softmax(tf.matmul(last, weight) + bias)


	prediction = RNN(x, weight, bias)

	# Define loss and optimizer
	loss_f = -tf.reduce_sum(y * tf.log(prediction))
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss_f)

	# Evaluate model
	correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	l_a = len(tr_features)
	# Initializing the variables
	init = tf.global_variables_initializer()
	import time
	a = time.time()
	with tf.Session() as session:
		session.run(init)
		
		for itr in range(training_iters):  
			for i in range(1,l_a/batch_size):  
				print 'running rrn',i
				batch_x = tr_features[i*batch_size:i*batch_size+batch_size]
				batch_y = tr_labels[i*batch_size:i*batch_size+batch_size]

				_, c = session.run([optimizer, loss_f],feed_dict={x: batch_x, y : batch_y})
					
				if epoch % display_step == 0:
					# Calculate batch accuracy
					acc = session.run(accuracy, feed_dict={x: batch_x, y: batch_y})
					# Calculate batch loss
					loss = session.run(loss_f, feed_dict={x: batch_x, y: batch_y})
					print "Iter " + str(epoch) + ", Minibatch Loss= " + \
						  "{:.6f}".format(loss) + ", Training Accuracy= " + \
						  "{:.5f}".format(acc)
		
		print('Test accuracy: ',round(session.run(accuracy, feed_dict={x: ts_features, y: ts_labels}) , 3))
	print time.time() - accuracy 