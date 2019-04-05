# Isaac Hogan
# 10188271
# Group 27
# COGS400
# 
# Group Project - Feed Forward Network for Current Rating Bias

import tensorflow as tf
import numpy as np
import csv, copy, random, math, sys

#reads first maxLines rows of a csv file (all if smaller)
def readData( file, maxLines ):
	print "\nreading data..."
	data = []
	with open(file, "r") as data_file:
		data_reader = csv.reader(data_file, delimiter=',')
		labels = next(data_reader)
		lineNum = 0
		for line in data_reader:
			lineNum += 1
			newline = map(float, line)
			data.append(newline)
		data.sort(key=lambda x: x[3])
		if lineNum >= maxLines:
			data = data[:maxLines]
	return data
	
#writes a list to a CSV file
def writeCSV( data, file ):
	#print "\nwriting CSV..."
	with open(file, "w") as data_file:
		for row in data:
			line = ""
			for element in row:
				line += str(element) + ","
			line = line[:-1] + "\n"
			data_file.write(line)

#prints a list to a file
def writeListToFile( list, file ):
	with open(file, "w") as data_file:
		data_file.write(str(list))
	
#extracts label column from list
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
	
#adds rating at time of review, remove movies with < cutoff reviews at time of review
def addTimeRating( data, cutoff ):
	print "\nadding rating at review time..."
	data.sort(key=lambda x: x[3])
	newData = []
	movieIDs = [int(x[1]) for x in data]
	movieIDs = dict.fromkeys(movieIDs)
	counter = 0
	for movie in movieIDs:
		movieIDs[movie] = [0,0]
	for row in data:
		#counter += 1
		#if counter % 1000000 == 0:
		#	print "row:", counter
		movie = row[1]
		if movieIDs[movie][0] >= cutoff:
			newData.append(row + [float(movieIDs[movie][1]) / float(movieIDs[movie][0])])
		movieIDs[movie][0] += 1
		movieIDs[movie][1] += row[2]
	return newData
	
#adds average movie ratings
def addAvgRating( data ):
	print "\nadding average rating..."
	newData = []
	movieIDs = [int(x[1]) for x in data]
	movieIDs = dict.fromkeys(movieIDs)
	counter = 0
	for movie in movieIDs:
		movieIDs[movie] = [0,0]
	for row in data:
		#counter += 1
		#if counter % 100000 == 0:
		#	print "row:", counter
		movie = row[1]
		movieIDs[movie][0] += 1
		movieIDs[movie][1] += row[2]
	counter = 0
	for row in data:
		#counter += 1
		#if counter % 1000000 == 0:
		#	print "row:", counter
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
	
#splits the data into test and training sets according 
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
	classif = tf.matmul(H1, W2) + B2
	
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
			print "epoch:", ep + 1
			for batchID in range(maxBatchID):
				batchRows = rows[(batchID * batchSize) : ((batchID + 1) * batchSize)]
				batchLabels = labels[(batchID * batchSize) : ((batchID + 1) * batchSize)]
				print str((100*batchID) // maxBatchID) + "%"
				sys.stdout.write("\033[F")
				for adr in range(batchSize):
					sess.run(train, feed_dict={row: [batchRows[adr]], label: [batchLabels[adr]]})
				#print "Squared_Error", sess.run(squaredError,feed_dict={row: [batchRows[adr]], label: [batchLabels[adr]]})
				#print "Loss", sess.run(loss,feed_dict={row: [batchRows[adr]], label: [batchLabels[adr]]})
				#print "Class", sess.run(classif,feed_dict={row: [batchRows[adr]], label: [batchLabels[adr]]})
				#print "Label", sess.run(label,feed_dict={row: [batchRows[adr]], label: [batchLabels[adr]]})
				#print "batch:", batchID + 1
		#print "calculating MSE..."
		model = sess.run(W1), sess.run(W2), sess.run(B1), sess.run(B2)
		#MSE, _ = testModel( labels, rows, model, hidden)	
		#sys.stdout.write("\033[F")
		#print "MSE:", MSE[0][0]
	return model
		
#returns all classifications as well as MSE
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
	classif = tf.matmul(H1, W2) + B2
	
	#loss and error functions
	squaredError = tf.square(label - classif)
	#loss = tf.reduce_sum(squaredError)
	
	totalError = 0
	results = []
	
	with tf.Session() as sess:
		sess.run( tf.global_variables_initializer() )
		for adr in range(len(rows)):
			if adr % 1000 == 0:
				print str((100*adr) // len(rows)) + "%"
				sys.stdout.write("\033[F")
			totalError += sess.run(squaredError, feed_dict={row: [rows[adr]], label: [labels[adr]], W1: weights[0], W2: weights[1], B1: weights[2], B2: weights[3]})
			results.append([ sess.run(label, feed_dict={row: [rows[adr]], label: [labels[adr]], W1: weights[0], W2: weights[1], B1: weights[2], B2: weights[3]}), sess.run(classif, feed_dict={row: [rows[adr]], label: [labels[adr]], W1: weights[0], W2: weights[1], B1: weights[2], B2: weights[3]})])
	print "       "
	MSE = totalError / len(rows)
	return MSE, results
			
# constants
filename = "./data/ratings_small.csv"
new_data = "./data/ratings_processed_out.csv"
results_1 = "./results/classification_m1.csv"
results_2 = "./results/classification_m2.csv"
model_1 = "./models/m1.txt"
model_2 = "./models/m2.txt"
hidden = 5
lrate = 0.01
batch_size = 1000 #batch processing not implemented, used for printing messages
epochs = 10
label_column = 2 #change this if location of label changes (ie. removeCol)
training_split = 0.2
data_size = 500000
movie_cutoff = 10

#for converting ratings.cvs
#ratings_data = readData( "../project_data/ratings.csv", 10000000 )
ratings_data = readData( filename, data_size )
ratings_data = addAvgRating( ratings_data )
ratings_data = addTimeRating( ratings_data, movie_cutoff )

print "\nRows of data (total):", len(ratings_data)

ratings_data =  removeCol(ratings_data, 0) #remove id
train, test = splitData( ratings_data, 0.2 ) #split train and test sets

print "Rows of data (train):", len(train)
print "Rows of data (test):", len(test)

#write the dataset to CSV file for reproducability
writeCSV( ratings_data, new_data )

#train and test model1
trainLabels, trainRows = splitLabels( label_column, train ) #train row/labels
trainRows = normalize(trainRows) #normalize train rows
model1 = trainModel( trainLabels, trainRows, hidden, lrate, batch_size, epochs ) 
testLabels, testRows = splitLabels( label_column, test ) #test row/labels
testRows = normalize(testRows) #normalize train rows
mse1, results1 = testModel( testLabels, testRows, model1, hidden )

#train and test model2
trainRows = removeCol( trainRows, (len(trainRows[0]) - 1) )
trainLabels, trainRows = splitLabels( label_column, train ) #train row/labels
model2 = trainModel( trainLabels, trainRows, hidden, lrate, batch_size, epochs )
mse2, results2 = testModel( testLabels, testRows, model2, hidden )

#write results
writeCSV( results1, results_1 )
writeCSV( results2, results_2 )
#writeCSV( ratings_data, "../project_data/ratings_avgs.csv" )

#write weights
writeListToFile( model1, model_1 )
writeListToFile( model2, model_2 )

#print mean errors
print "\nResults:\nmodel1 mean error:", math.sqrt(mse1), "\tmodel2 mean error:", math.sqrt(mse2)
			
			
			