# Isaac Hogan
# 10188271
# Group 27
# COGS400
# 
# Group Project - Feed Forward Network for Current Rating Bias

import tensorflow as tf
import numpy as np
import csv, copy, random, math

def readData( file, maxLines ):
	print "\nreading data..."
	data = []
	with open(file, "r") as data_file:
		data_reader = csv.reader(data_file, delimiter=',')
		labels = next(data_reader)
		lineNum = 0
		for line in data_reader:
			lineNum += 1
			if lineNum <= maxLines:
				newline = map(float, line)
				data.append(newline)
			else:
				break
	return data

	
def writeCSV( data, file ):
	print "\nwriting CSV..."
	data.sort(key=lambda x: x[0])
	with open(file, "w") as data_file:
		for row in data:
			line = ""
			for element in row:
				line += str(element) + ","
			line = line[:-1] + "\n"
			data_file.write(line)
	
def splitLabels( labelColumn, data ):
	#print "\nspliting data..."
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
	
def addTimeRating( data ):
	print "\nadding rating at review time..."
	data.sort(key=lambda x: x[3])
	newData = []
	movieIDs = [int(x[1]) for x in data]
	movieIDs = dict.fromkeys(movieIDs)
	counter = 0
	for movie in movieIDs:
		movieIDs[movie] = [0,0]
	for row in data:
		counter += 1
		if counter % 100000 == 0:
			print "row:", counter
		movie = row[1]
		if movieIDs[movie][0] == 0:
			newData.append(row + [0])
		else:
			newData.append(row + [float(movieIDs[movie][1]) / float(movieIDs[movie][0])])
		movieIDs[movie][0] += 1
		movieIDs[movie][1] += row[2]
	return newData
		
def addAvgRating( data ):
	print "\nadding average rating..."
	newData = []
	movieIDs = [int(x[1]) for x in data]
	movieIDs = dict.fromkeys(movieIDs)
	counter = 0
	for movie in movieIDs:
		movieIDs[movie] = [0,0]
	for row in data:
		counter += 1
		if counter % 100000 == 0:
			print "row:", counter
		movie = row[1]
		movieIDs[movie][0] += 1
		movieIDs[movie][1] += row[2]
	counter = 0
	for row in data:
		counter += 1
		if counter % 100000 == 0:
			print "row:", counter
		movie = row[1]
		if movieIDs[movie][0] == 0:
			newData.append(row + [0])
		else:
			newData.append(row + [float(movieIDs[movie][1]) / float(movieIDs[movie][0])])
	return newData	
	
#removes a column from a dataset
def removeCol(matrix, address):
	newMatrix = []
	for row in range(len(matrix)):
		newMatrix.append([])
		for col in range(len(matrix[row])):
			if col != address:
				newMatrix[row].append(matrix[row][col])
	return newMatrix
	
def splitData( data, testProb ):
	trainData, testData = [], []
	dataCopy = copy.deepcopy(data)
	for row in data:
		if random.random() < testProb:
			testData.append(row)
		else:
			trainData.append(row)
	return trainData, testData
	
#Usage = trainModel( <lables_list>, <data_rows_list>, <#_hidden_nodes>, <learning_rate>, <#_training_epocs> )
#Returns: python list of all weights 	
def trainModel( labels, rows, hidden, learnRate, batchSize, epochs ):
	
	print "\ntraining model..."
	
	#input data and labels
	row = tf.placeholder(shape=(1,len(rows[0])), dtype=tf.float64, name='row')
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
				print "Squared_Error", sess.run(squaredError,feed_dict={row: [batchRows[adr]], label: [batchLabels[adr]]})
				print "Loss", sess.run(loss,feed_dict={row: [batchRows[adr]], label: [batchLabels[adr]]})
				print "Class", sess.run(classif,feed_dict={row: [batchRows[adr]], label: [batchLabels[adr]]})
				print "Label", sess.run(label,feed_dict={row: [batchRows[adr]], label: [batchLabels[adr]]})
				#print "batch:", batchID + 1
			model = sess.run(W1), sess.run(W2), sess.run(B1), sess.run(B2)
			MSE = testModel( labels, rows, model, hidden)	
			print "epoch:", ep + 1, "MSE:", MSE
	return model
		
def testModel( labels, rows, weights, hidden ):

	#input and label data
	row = tf.placeholder(shape=(1,len(rows[0])), dtype=tf.float64, name='row')
	label = tf.placeholder(shape=(1,1), dtype=tf.float64, name='label')
	
	#weights
	W1 = tf.placeholder(shape=(len(rows[0]), hidden), dtype=tf.float64, name='W1')
	W2 = tf.placeholder(shape=(hidden, len(labels[0])), dtype=tf.float64, name='W2')
	
	#bias
	B1 = tf.placeholder(shape=(hidden), dtype=tf.float64, name='B1')
	B2 = tf.placeholder(shape=(len(labels[0])), dtype=tf.float64, name='B2')
	
	#define net flow
	H1 = tf.sigmoid(tf.matmul(row, W1) + B1)
	classif = tf.sigmoid(tf.matmul(H1, W2) + B2)
	
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
lrate = 0.01
batch_size = 1000
epochs = 1
label_column = 2
training_split = 0.2

#for converting ratings.cvs
#ratings_data = readData( "../project_data/ratings.csv", 10000000 )
ratings_data = readData( filename, 10000000 )
ratings_data = addAvgRating( ratings_data )
ratings_data = addTimeRating( ratings_data )
#writeCSV( ratings_data, "../project_data/ratings_avgs.csv" )

ratings_data =  removeCol(ratings_data, 0) #remove id
train, test = splitData( ratings_data, 0.2 ) #split train and test sets

trainLabels, trainRows = splitLabels( label_column, train ) #train row/labels
trainRows = normalize(trainRows) #normalize train rows
model1 = trainModel( trainLabels, trainRows, hidden, lrate, batch_size, epochs ) 
testLabels, testRows = splitLabels( label_column, test ) #test row/labels
mse1 = testModel( testLabels, testRows, model1, hidden )

trainRows = removeCol( trainRows, (len(trainRows[0]) - 1) )
trainLabels, trainRows = splitLabels( label_column, train ) #train row/labels
model2 = trainModel( trainLabels, trainRows, hidden, lrate, batch_size, epochs )
mse2 = testModel( testLabels, testRows, model2, hidden )

#print trainRows
print "model 1 mean error:", math.sqrt(mse1), "\tmodel 2 mean error:", math.sqrt(mse2)
			
			
			