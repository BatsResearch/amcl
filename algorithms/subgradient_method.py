import pulp as lp
import numpy as np
import scipy
import itertools
import math
import operator as op
from statistics import mean
from scipy.optimize import linprog
import random
from numpy import linalg as LA
from scipy.optimize import approx_fprime

##################################

# Define your loss functions here
# The first argument y_1 represents the probability vector of the labels of the item (true labels)
# The second argument y_2 represents the guess of the label of the item (e.g., output of your model)
def Brier_loss_linear(y_1, y_2):
    # Check if y_1 and y_2 have the same length
    C = len(y_1)
    if C != len(y_2):
        raise NameError('Computing loss on vectors of different lengths')

    # Compute the expected loss
    x = 0
    for i in range(C):
        cum = 0
        for j in range(C):
            if i != j:
                cum += y_2[j]**2
        x += y_1[i]*((1 - y_2[i])**2 + cum)/2
    return x


def cross_entropy_linear(y_1, y_2):
    # Check if y_1 and y_2 have the same length
    C = len(y_1)
    if C!= len(y_2):
        print(len(y_2), len(y_1))
        raise NameError('Computing loss on vectors of different lengths')

    # Clipping values for numerical issues
    y_2 = np.clip(y_2, 1e-5, None)   
    #Compute and return the loss
    return np.sum(-y_1 * np.log(y_2))

##################################
# This function builds the linear constraints that represents the
# feasible set of labeling based on the unlabeled data
# and the labeled data. The output of this functions are three
# elements: constraint_matrix, constraint_vector, constraint_sign
# The i-th element of each of these three vectors is respectively
# the coefficients, the constant term, and the kind of the inequality
# ( <=, ==, >=) of the i-th linear constraint of the feasible set.
# The variables of these linear constraints are M*C, where M
# is the number of unlabeled data points and C is the number
# of classes. The (j*C + c)-th variable represents the probability
# that the j-th unlabeled item belongs to class c.
def compute_constraints_with_loss( lf, output_labelers_unlabeled, output_labelers_labeled, true_labels): # lf == loss_function
    constraint_matrix = [] # The i-th element is a vector that represents the coefficient of the i-th linear constraint
    constraint_vector = [] # The i-th element is the constant term of the i-th linear constraint
    constraint_sign = [] # The i-th element represents the sign of the inequality of the i-th linear constraint
                         # i.e., -1 is <=, 0 is ==, 1 is >=

    N = len(output_labelers_unlabeled) # Number of weak classifiers
    M = len(output_labelers_unlabeled[0]) # Number of unlabeled data points
    C = len(output_labelers_unlabeled[0][0]) # Number of classes
    Ml = len(output_labelers_labeled[0]) # Number of labeled data points
    print(N,M,C,Ml)

    
    if Ml != len(true_labels):
        raise NameError('Labeled data points and label sizes are different')

    # For each weak classifier
    for i in range(N):
        # Compute the expected error over the labeled data for each weak classifier
        error = 0
        for j in range(Ml):
            error += lf(true_labels[j],output_labelers_labeled[i][j])
            # print(error)
        error = error/Ml

        # Compute the coefficient of the linear constraint based on the 
        build_coefficients = []
        for j in range(M):
            for c in range(C):
                e = np.zeros(C)
                e[c] = 1  # e is a vector of only zeros with a one in position j
                build_coefficients.append(  lf( e, output_labelers_unlabeled[i][j])/M) # For later, check the division here
        
        # Bounds: risk of a labeler must be within error+-offset
        ####
        delta = 0.1 # Between 0 and 1. Probability of the true labeling to NOT belong to the feasible set.
        scaling_factor = 0.1 # Direct computation of the offset could yield large values if M or Ml is small.
                             # This number can be used to scale the offset if it is too large 
        offset = scaling_factor * np.sqrt( (Ml + M)*np.log(4*N/delta)/(2*(Ml*M)))
        offset = 0 # Uncomment this line 
                   # if you do not want to have a offset. This could be better in practice if
                   # the number of labeled data and labeled data is very large
        ####

        # Add less or equal constraint
        constraint_matrix.append( build_coefficients)
        constraint_sign.append(-1)
        constraint_vector.append(error + offset)  
        # Add greater or equal constraint
        constraint_matrix.append( build_coefficients)
        constraint_sign.append(1)
        constraint_vector.append(error - offset)  

    # Add constraints for the sum of the label probabilities for each item to sum to 1
    for j in range(M):
        tmp = np.zeros(M*C)
        for i in range(j*C, (j+1)*C):
            tmp[i] = 1
        constraint_matrix.append(tmp)
        constraint_sign.append(0)
        constraint_vector.append(1)
    return constraint_matrix,constraint_vector,constraint_sign

##################################
# This function computes the worst-case error on the unlabeled data of a 
# classifier represented with cost_vector with respect to an adversarial choice
# of the feasible labeling of the unlabeled data (i.e., it solves the max of the minimax).
# cost_vector[j*C + c] is the error of the classifier on the j-th item if its true label was c.
def solveLPGivenCost(constraint_matrix,constraint_vector,constraint_sign,cost_vector):
    D = len(constraint_matrix[0]) # D = C*M (number of variables of the linear program)
    nc = len(constraint_matrix) # Number of constraints
    obj = cost_vector  # Objective function of the linear program
    # We build the constraints of the linear program using constraint_matrix, constraint_vector and constraint_sign
    lsh_ineq = []
    rsh_ineq = []
    lsh_eq = []
    rsh_eq = []
    for i in range(nc):
        if constraint_sign[i] == 0:
            # Equality constraints
            lsh_eq.append(constraint_matrix[i])
            rsh_eq.append(constraint_vector[i])
        if constraint_sign[i] != 0:
            # Inequality constraints
            lsh_ineq.append( np.multiply(constraint_matrix[i], -constraint_sign[i]))
            rsh_ineq.append( np.multiply(constraint_vector[i], -constraint_sign[i]))
    # Solve the linaer program
    opt = linprog(c = obj, A_ub = lsh_ineq, b_ub = rsh_ineq, A_eq = lsh_eq, b_eq = rsh_eq,  method = "interior-point",options = {"sparse": False,"presolve": True, "tol": 1e-12})
    return opt.x, opt.fun

##################################
# Models implementation
# linear_combination_labeler => linear combination of the weak classifiers
# logistic_regression => multinomial logistic regression

# X is a matrix representing the output of each weak classifier
# on the unlabeled data. This function computes the output of the 
# classifier that combines for each item the output of the weak 
# classifiers according to the weighting theta. The weights theta must sum to 1
def linear_combination_labeler(theta,X):
    num_wls, num_classes = np.shape(X)
    cum = np.zeros(num_classes)
    for i in range(num_wls):
        cum = np.add(cum, np.multiply(X[i],theta[i]))
    return cum

# X is matrix with a feature representation
# of the unlabeled data. This function computes the output of
# the multinomial logistic regression according to the weights theta.
def logistic_regression(theta, X):
    ret = np.exp(np.dot(X, theta))
    ret /= np.sum(ret)
    # ret = ret/cum
    return ret

##################################
# Projection methods used during gradient descent


# Project a vector to the simplex
# Code - implementation taken from https://gist.github.com/mblondel/6f3b7aaad90606b98f71
def projectToSimplex(v):
    
    v = np.array(v)
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

# Project a vector inside a sphere of radius maxDist
def projectToBall(v, maxDist = 100):
    v = np.array(v)
    c = min(maxDist/np.linalg.norm(v),1)
    return v/c


##################################
# Gradient computation for different models-loss function
# h is the prediction model

# Gradient computation for multinomial logistic regression with cross-entropy loss
def computeGradientLR(theta, X, Y, h):
    M = len(X)
    Y = np.array(Y)
    Y_predict = np.array([ h(theta, X[i]) for i in range(M)])
    Y_diff = Y_predict - Y
    grads = np.array([np.kron(Y_diff[i],X[i]) for i in range(M)])
    grad = np.average(grads,axis=0)
    return grad

# Gradient computation for convex combination of weak classifiers with Brier Loss
def computeGradientCOMB(theta, X, Y, h):
    X = np.array(X)
    Y = np.array(Y)
    pred = np.array([ h(theta, X[:,j]) for j in range(len(Y))])
    grad = np.array([2*np.dot(X[:,j],(pred[j]-Y[j])) for j in range(len(Y))])
    grad = np.average(grad,axis=0)
    return grad


#####################################
# * X_unlabeled => unlabeled data
# * constraint_matrix, constraint_vector, constraint_sign => constraints of the feasible set
#   as computed from the method compute_constraints_with_loss
# * lf => loss function
# * h => prediction model (the first argument is the weights of the model)
# * proj_function => projection method for the weights of the prediction model
# * initial_theta => initial weights of the prediction model
# * iteration => number of iterations
# * step_size => step size for the subgradient descent method
# * N => number of weak classifiers
# * M => number of unlabeled data points
# * C => number of classes
# * lr => flag: true if we are using multinomial logistic regression - false if we are using
#   convex combination of weak classifiers
def subGradientMethod(X_unlabeled, constraint_matrix,constraint_vector,constraint_sign, 
    lf, h, proj_function, initial_theta, iteration, step_size, N, M, C, lr=False):

    # These functions are defined for convenience
    #  on the current labeling of the unlabeled data
    # according to the loss function lf

    # Evaluation for convex combination of weak classifier
    def eval_theta(th):
        cum = 0
        for j in range(M):
            cum += lf(new_y[j], h(th,X_unlabeled[:,j]))

        return cum/M
    
    # Evaluation for multinomial logistic regression
    def eval_lr(th):
        cum = 0
        for j in range(M):
            cum += lf(new_y[j], h(th,X_unlabeled[j]))
        return cum/M
    # Different implementation
    def eval_lr_iterative(th):
        cum = 0
        # for j in range(M):
            # cum += lf(new_y[j], h(th.reshape(-1, C),X_unlabeled[j]))
        # cum = np.sum(lf(new_y, h(th, X_unlabeled)))
        cum = np.sum(lf(new_y, h(th.reshape(-1, C), X_unlabeled)))
        return cum/M
    ##

    # Current value of the minimax
    best_val = 10e10 # Initialized to a very high value
    theta = initial_theta # Weights of the model
    cost= [] # Cost vector, cost[j*C + c] is the loss of current model to item j if its true label was c
    for j in range(M):
        for c in range(C):
            e = np.zeros(C)
            e[c] = 1
            if lr:
                cost.append(-lf(e, h(theta, X_unlabeled[j])) / M)
            else:
                cost.append(-lf(e, h(theta,X_unlabeled[:,j]))/M)

    # Find labeling that maximizes the error
    new_y,_ = solveLPGivenCost(constraint_matrix,constraint_vector,constraint_sign,cost)
    new_y = np.array([new_y[i*C : (i + 1) * C] for i in range(M)])

    if not lr:
        print("STARTING PARAMETERS: " + str(theta))

    # Subgradient method core implementation
    # Iterate this process for iteration
    for t in range(iteration):
        # Compute subgradient with respect to theta and the current labeling
        # of the unlabeled data
        if lr:
            grad = computeGradientLR(theta,X_unlabeled,new_y,h)
            theta = theta.T.flatten()
        else:
            grad = computeGradientCOMB(theta, X_unlabeled, new_y,  h)

        # Gradient descent step
        theta = np.add(theta, np.multiply(grad,-step_size))

        # Projection step
        theta = proj_function(theta)

        # If multinomial logistic regression, convert weights in matrix form
        if lr:
            # convert back to matrix form
            theta = theta.reshape((C, -1)).T
        
        # Re-compute the worst-case error after the gradient step
        cost = []
        for j in range(M):
            for c in range(C):
                e = np.zeros(C)
                e[c] = 1
                if lr:
                    cost.append(-lf(e, h(theta, X_unlabeled[j])) / M)
                else:
                    cost.append(-lf(e, h(theta,X_unlabeled[:,j]))/M)
        # Find labeling that maximizes the error
        new_y, obj = solveLPGivenCost(constraint_matrix,constraint_vector,constraint_sign,cost)
        new_y = np.array([new_y[i*C : (i + 1) * C] for i in range(M)])

        # Evaluate the current model with respect to the worst-case error
        if not lr:
            current_eval = eval_theta(theta)
        if lr:
            current_eval = eval_lr(theta)
        # If the current model is better, update the best model found
        if (current_eval < best_val):
            best_theta = theta
            best_val = current_eval

        # Debug lines
        if t % 50 == 0:
            # if t % 100 == 0:
                # print("Theta: ", theta )

            vals = []
            if lr:
                M, C = np.shape(X_unlabeled)
            else:
                _, M, C = np.shape(X_unlabeled)
            totals = np.zeros((M, C))
            for i, val in enumerate(best_theta):
                for j, vote in enumerate(X_unlabeled[i]):
                    totals[j] += val * vote

            for i in range(len(new_y)):
                vals.append(Brier_loss_linear(new_y[i], totals[i]))
            print("Bound: " + str(np.mean(vals)))

    # Debug lines
    if not lr:
        print("ENDING PARAMETERS: " + str(best_theta))

	# Return the best model found during the subgradient method execution
    return best_theta