import numpy as np 
from sklearn.model_selection import train_test_split
import random
import itertools
import algorithms.subgradient_method as SG
import algorithms.max_likelihood as ML

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
		vals.append(SG.Brier_loss_linear(y[i], y_pred[i]))
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
		p = SG.logistic_regression(theta, d)
		preds.append(np.argmax(p))
		probs.append(p)
	
	probs = np.array(probs)
	preds = np.array(preds)

	briar_loss = compute_avg_briar(labels, probs, C)
	return np.mean(preds == labels), briar_loss

def run_test(data, labels, votes, ratio=0.2, model="weighted-vote"):
	'''
	Function to run experiments on the AwA2 dataset
	
	Args:
	data - 2D array containing the dataset
	labels - 1D array containing labels for the data
	votes - 3D array containing multi-class votes of weak supervision sources on data
	ratio - ratio of labeled data
	model - method to aggregate results (weighted vote or logistic regression)
	'''

	# splitting data into an train (labeled) and test (unlabeled) set
	train_indices, test_indices = train_test_split(range(len(labels)), test_size=1-ratio, stratify=labels)
	train_data, train_labels = data[train_indices], labels[train_indices]
	test_data, test_labels = data[test_indices], labels[test_indices]
	train_votes, test_votes = votes[:,train_indices], votes[:,test_indices]

	# convert data to dimensions of # examples x features
	# train_data = np.transpose(train_data, [1, 0, 2])
	# test_data = np.transpose(test_data, [1, 0, 2])

	N = np.shape(train_votes)[0] # number of weak supervision sources
	C = np.shape(train_votes)[2] # number of classes

	# Unlabeled data and Labeled data
	num_lab = len(train_labels)
	num_unlab = len(test_labels)
	print("Unlab: " + str(num_unlab) + " | Lab: " + str(num_lab))

	# labels converted into one hots
	train_labels = np.eye(C)[train_labels - 1]
	test_labels = np.eye(C)[test_labels - 1]

	# SET algorithm parameters here
	eps = 0.3
	L = 2 * np.sqrt(N + 1)
	squared_diam = 2
	T = int(np.ceil(L*L*squared_diam/(eps*eps)))
	T = 1000
	h = eps/(L*L)

	if model == "logistic-regression":
		import warnings
		warnings.filterwarnings('ignore')
		constraint_matrix, constraint_vector, constraint_sign = SG.compute_constraints_with_loss(SG.cross_entropy_linear, 
																								test_votes, 
																								train_votes, 
																								train_labels)
		# transforming data w/ Resnet
		initial_theta = np.random.normal(0, 0.1, (np.shape(test_data)[1], C))
		model_theta = SG.subGradientMethod(test_data, constraint_matrix, constraint_vector, 
										constraint_sign, SG.cross_entropy_linear, SG.logistic_regression, 
										SG.projectToBall,initial_theta, 
										T, h, N, num_unlab, C, lr=True)

		c = eval_lr(test_votes, test_labels, model_theta, C)
		print("Logistic regression: " + str(c)) # acc ,  briar loss


	if model == "weighted-vote":
		constraint_matrix, constraint_vector, constraint_sign = SG.compute_constraints_with_loss(SG.Brier_loss_linear, 
																								test_votes, 
																								train_votes, 
																								train_labels)
		initial_theta = np.array([1 / N for i in range(N)])
		model_theta = SG.subGradientMethod(test_votes, constraint_matrix, constraint_vector, 
										constraint_sign, SG.Brier_loss_linear, SG.linear_combination_labeler, 
										SG.projectToSimplex, initial_theta, 
										T, h, N, num_unlab, C)
		
		# evaluate learned model
		c = eval_weighted(test_votes, test_labels, model_theta)
		print("Weighted vote: " + str(c)) # acc ,  briar loss

	if model == "individual":		
		results_dict = {}

		for i in range(N):
			wl_preds = np.argmax(test_votes[i], axis=1)
			results_dict[i] = (np.mean(wl_preds == np.argmax(test_labels)), compute_avg_briar(test_labels, test_votes[i], C))

		# sort list by lowest 0-1 acc
		sor_res = sorted(results_dict.items(), key=lambda x: -x[1][0])
		for i in range(3):
			print(sor_res[i])
	
if __name__ == "__main__":
	
	# fixing seed
	seed = 0
	np.random.seed(seed)
	random.seed(0)

	# running weighted vote with 20% of data as labeled
	model = "weighted-vote" # "logistic-regression"
	ratio_labeled = 0.2
	
	# test example learns model on weak supervision sources not raw pixels
	data = np.load("example_data/fake_data.npy")
	votes = np.load("example_data/votes.npy")
	labels = np.load("example_data/labels.npy")

	# running experiments
	run_test(data, labels, votes, ratio_labeled, model)