import tensorflow as tf
import numpy as np
import csv

def readData( file ):
	print "\nreading data..."
	data = []
	with open(file, "r") as data_file:
		data_reader = csv.reader(data_file, delimiter=',')
		labels = next(data_reader)
		for line in data_reader:
			newline = map(float, line)
			data.append(newline)	
	return data
	
def splitData( labelColumn, data ):
	print "\nspliting data..."
	label, rows = [], []
	for row in range(len(data)):
		label.append([data[row][labelColumn - 1]])
		rows.append(data[row][:(labelColumn - 1)] + data[row][-(len(data[row]) - (labelColumn)):])
	return label, rows
	
#normalize data values	
def normalize(data):
	max = data[0][:]
	min = data[0][:]
	for entry in data:
		for val in range(len(entry)):
			if entry[val] > max[val]:
				max[val] = entry[val]
			elif entry[val] < min[val]:
				min[val] = entry[val]
	for entry in range(len(data)):
		for val in range(len(data[entry])):
			if max[val] != min[val]:
				data[entry][val] = (data[entry][val] - min[val]) / (max[val] - min[val])
			else:
				data[entry][val] = 0
	return data
	
#Usage = trainModel( <lables_list>, <data_rows_list>, <#_hidden_nodes>, <learning_rate>, <#_training_epocs> )
#Returns: python list of all weights 	
def trainModel( labels, rows, hidden, learnRate, batchSize, epochs ):
	
	print "\ntraining model..."
	
	#input data and labels
	row = tf.placeholder(shape=(1,3), dtype=tf.float64, name='row')
	label = tf.placeholder(shape=(1,1), dtype=tf.float64, name='label')
	
	#weights
	W1 = tf.Variable(np.random.rand(len(rows[0]), hidden), dtype=tf.float64)
	W2 = tf.Variable(np.random.rand(hidden, len(labels[0])), dtype=tf.float64)
	
	#bias
	B1 = tf.Variable(np.random.rand(hidden), dtype=tf.float64)
	B2 = tf.Variable(np.random.rand(len(labels[0])), dtype=tf.float64)
	
	#define net flow
	H1 = tf.sigmoid(tf.matmul(row, W1) + B1)
	classif = tf.sigmoid(tf.matmul(H1, W2) + B2)
	
	#loss and error functions
	squaredError = tf.square(label - classif)
	loss = tf.reduce_sum(squaredError)
	
	#training operation
	optimizer = tf.train.GradientDescentOptimizer(learnRate)
	train = optimizer.minimize(loss)
	
	#traning session
	with tf.Session() as sess:
		sess.run( tf.global_variables_initializer() )
		maxBatchID = len(rows) // batchSize
		for ep in range(epochs):
			for batchID in range(maxBatchID):
				batchRows = rows[(batchID * batchSize) : ((batchID + 1) * batchSize)]
				batchLabels = labels[(batchID * batchSize) : ((batchID + 1) * batchSize)]
				for adr in range(batchSize):
					sess.run(train, feed_dict={row: [batchRows[adr]], label: [batchLabels[adr]]})
				#print "batch:", batchID + 1
			model = sess.run(W1), sess.run(W2), sess.run(B1), sess.run(B2)
			MSE = testModel( labels, rows, model, hidden)	
			print "epoch:", ep + 1, "MSE:", MSE
	return model
		
def testModel( labels, rows, weights, hidden ):

	#input and label data
	row = tf.placeholder(shape=(1,3), dtype=tf.float64, name='row')
	label = tf.placeholder(shape=(1,1), dtype=tf.float64, name='label')
	
	#weights
	W1 = tf.placeholder(shape=(len(rows[0]), hidden), dtype=tf.float64, name='W1')
	W2 = tf.placeholder(shape=(hidden, len(labels[0])), dtype=tf.float64, name='W2')
	
	#bias
	B1 = tf.placeholder(shape=(hidden), dtype=tf.float64, name='B1')
	B2 = tf.placeholder(shape=(len(labels[0])), dtype=tf.float64, name='B2')
	
	#define net flow
	H1 = tf.sigmoid(tf.matmul(row, W1) + B1)
	classif = tf.sigmoid(tf.matmul(H1, W2))
	
	#loss and error functions
	squaredError = tf.square(label - classif)
	#loss = tf.reduce_sum(squaredError)
	
	totalError = 0
	
	with tf.Session() as sess:
		sess.run( tf.global_variables_initializer() )
		for adr in range(len(rows)):
			totalError += sess.run(squaredError, feed_dict={row: [rows[adr]], label: [labels[adr]], W1: weights[0], W2: weights[1], B1: weights[2], B2: weights[3]})
	MSE = totalError / len(rows)
	
	return MSE
			
# constants
filename = "./data/ratings_small.csv"
hidden = 4
lrate = 1
batch_size = 1000
epochs = 1
label_column = 3

ratings_data = readData( filename )
labels, rows = splitData( label_column, ratings_data )
rows = normalize(rows)
model = trainModel( labels, rows, hidden, lrate, batch_size, epochs ) 
sse = testModel( labels, rows, model, hidden)
print sse
			
			
			
			
			