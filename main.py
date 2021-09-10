import numpy as np 
from sklearn.model_selection import train_test_split
import random
import itertools
import algorithms.subgradient_method as SG
import algorithms.max_likelihood as ML
import algorithms.util as util

def compute_avg_briar(y, y_pred, C):
	'''
	Function to compute the average briar loss over each example

	Args:
	y - true labels (one-hots)
	y_pred - prediction from the model (probability distribution)
	C - number of classes
	'''

	vals = []

	for i in range(len(y)):
		vals.append(util.Brier_loss_linear(y[i], y_pred[i]))
	return np.mean(vals)

def eval_weighted(votes, labels, theta):
	'''
	Function to compute the accuracy of a weighted combination of labelers
	
	Args:
	votes - weak supervision source outputs
	labels - one hot labels
	theta - the weighting given to each weak supervision source
	'''
	N, M, C = np.shape(votes)
	totals = np.zeros((M, C))
	for i, val in enumerate(theta):
		for j, vote in enumerate(votes[i]):
			totals[j] += val * vote

	preds = np.argmax(totals, axis=1)
	briar_loss = compute_avg_briar(labels, totals, C)
	return np.mean(preds == np.argmax(labels, axis=1)), briar_loss

def eval_majority(votes, sub):
	'''
	Function to evaluate the majority vote of a subset for binary tasks
	
	Args:
	votes - weak supervision source outputs
	sub - the subset of weak supervision sources to contribute to the majority vote
	'''

	N, M, C = np.shape(votes)
	probs = np.zeros((M, C))

	for i in sub:
		probs += votes[i]
	
	return probs / len(sub), np.argmax(probs, axis=1)

def eval_lr(data, labels, theta, C):
	'''
	Function to evaluate a logistic regression model 

	Args:
	data - the data to evaluate the logreg model on
	labels - one hot labels
	theta - the weights for the logreg model
	C - the number of target classes
	'''

	probs = []
	preds = []

	# reshaping data
	for i, d in enumerate(data):
		p = util.logistic_regression(theta, d)
		preds.append(np.argmax(p))
		probs.append(p)
	
	probs = np.array(probs)
	preds = np.array(preds)

	briar_loss = compute_avg_briar(labels, probs, C)
	return np.mean(preds == labels), briar_loss

def run_test(train_votes, train_labels, test_votes, test_labels):
	'''
	Function to run experiments on the AwA2 dataset (for weighted vote model)
	'''

	# splitting data into an train (labeled) and test (unlabeled) set
	N = 5 # number of weak supervision sources
	C = 5 # number of classes

	# Unlabeled data and Labeled data
	num_lab = len(train_labels)
	num_unlab = len(test_labels)
	print("Unlab: " + str(num_unlab) + " | Lab: " + str(num_lab))

	print("Individual weak supervision sources test accuracies")
	num_wls = len(l_votes[0])
	for i in range(num_wls):
		preds = l_votes[:,i]
		acc = np.mean(preds == l_labels)
		print("WL %d Acc: %f" % (i, acc))

	train_votes = np.eye(5)[train_votes]
	test_votes = np.eye(5)[test_votes]
	train_votes = np.transpose(train_votes, (1, 0, 2))
	test_votes = np.transpose(test_votes, (1, 0, 2))

	train_labels = np.eye(5)[train_labels]
	test_labels = np.eye(5)[test_labels]

	# SET algorithm parameters here
	eps = 0.3
	L = 2 * np.sqrt(N + 1)
	squared_diam = 2
	T = int(np.ceil(L*L*squared_diam/(eps*eps)))
	T = 500
	h = eps/(L*L)

	constraint_matrix, constraint_vector, constraint_sign = SG.compute_constraints_with_loss(util.Brier_loss_linear, 
																							test_votes, 
																							train_votes, 
																							train_labels)
	initial_theta = np.array([1 / N for i in range(N)])
	model_theta = SG.subGradientMethod(test_votes, constraint_matrix, constraint_vector, 
									constraint_sign, util.Brier_loss_linear, util.linear_combination_labeler, 
									util.projectToSimplex, initial_theta, 
									T, h, N, num_unlab, C)
	
	# evaluate learned model
	c = eval_weighted(test_votes, test_labels, model_theta)
	print("Weighted vote: " + str(c)) # acc ,  briar loss
	
if __name__ == "__main__":
	
	# fixing seed
	seed = 0
	np.random.seed(seed)
	random.seed(0)

	# running weighted vote with 20% of data as labeled
	model = "weighted-vote" # "logistic-regression"
	ratio_labeled = 0.2
	
	# test example learns model on weak supervision sources (cifar test)
	l_votes = np.load("example_data/l_votes.npy")
	l_labels = np.load("example_data/l_labels.npy")
	ul_votes = np.load("example_data/ul_votes.npy")
	ul_labels = np.load("example_data/ul_labels.npy")


	# running experiments
	run_test(l_votes, l_labels, ul_votes, ul_labels)