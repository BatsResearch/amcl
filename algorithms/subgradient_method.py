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

# Define your loss functions here
def Brier_loss_linear(y_1, y_2):
    # y_1 is the probability vector of a label occuring (true labels)
    # y_2 is the guess

    # print(np.shape(y_1), np.shape(y_2))

    C = len(y_1)
    if C != len(y_2):
        raise NameError('Computing loss on vectors of different lengths')
    x = 0
    for i in range(C):
        cum = 0
        for j in range(C):
            if i != j:
                cum += y_2[j]**2
        x += y_1[i]*((1 - y_2[i])**2 + cum)/2
    return x

def cross_entropy_linear(y_1, y_2):
    # y_1 is the true label
    # y_2 is the output of the classifier
    C = len(y_1)
    if C!= len(y_2):
        print(len(y_2), len(y_1))
        raise NameError('Computing loss on vectors of different lengths')
    x = 0
    # for i in range(C):
        # x += -y_1[i]*np.log(y_2[i])
    # return x

    # clipping value
    y_2 = np.clip(y_2, 1e-5, None)   
    return np.sum(-y_1 * np.log(y_2))

# Compute constraints using a loss function and labeled data
def compute_constraints_with_loss( lf, output_labelers_unlabeled, output_labelers_labeled, true_labels): # lf == loss_function
    constraint_matrix = []
    constraint_vector = []
    constraint_sign = []

    N = len(output_labelers_unlabeled) # Number of weak classifiers
    M = len(output_labelers_unlabeled[0]) # Number of unlabeled data points
    C = len(output_labelers_unlabeled[0][0]) # Number of classes
    Ml = len(output_labelers_labeled[0]) # Number of labeled data points
    print(N,M,C,Ml)

    if Ml != len(true_labels):
        raise NameError('Labeled data points and label sizes are different')

    for i in range(N):
        error = 0
        for j in range(Ml):
            error += lf(true_labels[j],output_labelers_labeled[i][j])
            # print(error)
        error = error/Ml

        build_coefficients = []
        for j in range(M):
            for c in range(C):
                e = np.zeros(C)
                e[c] = 1  # e is a vector of only zeros with a one in position j
                build_coefficients.append(  lf( e, output_labelers_unlabeled[i][j])/M) # For later, check the division here
        
        # Bounds: risk of a labeler must be within error+-offset
        ####
        delta = 0.4
        # delta = 0.1
        offset = 0.1 * np.sqrt( (Ml + M)*np.log(4*N/delta)/(2*(Ml*M)))
        ####

        # Less or equal constraint
        constraint_matrix.append( build_coefficients)
        constraint_sign.append(-1)
        constraint_vector.append(error + offset)  
        # Greater or equal constraint
        constraint_matrix.append( build_coefficients)
        constraint_sign.append(1)
        constraint_vector.append(error - offset)  

    # Add constraints for probabilities summing to 1
    for j in range(M):
        tmp = np.zeros(M*C)
        for i in range(j*C, (j+1)*C):
            tmp[i] = 1
        constraint_matrix.append(tmp)
        constraint_sign.append(0)
        constraint_vector.append(1)
    return constraint_matrix,constraint_vector,constraint_sign

def solveLPGivenCost(constraint_matrix,constraint_vector,constraint_sign,cost_vector):
    D = len(constraint_matrix[0])
    nc = len(constraint_matrix)
    obj = cost_vector  
    lsh_ineq = []
    rsh_ineq = []
    lsh_eq = []
    rsh_eq = []
    bnd = []
   # for i in range(D):
   #     bnd.append((0,float("inf")))
    

    # print(np.shape(constraint_matrix), np.shape(cost_vector))

    for i in range(nc):
        if constraint_sign[i] == 0:
            lsh_eq.append(constraint_matrix[i])
            rsh_eq.append(constraint_vector[i])
        if constraint_sign[i] != 0:
            lsh_ineq.append( np.multiply(constraint_matrix[i], -constraint_sign[i]))
            rsh_ineq.append( np.multiply(constraint_vector[i], -constraint_sign[i]))

    opt = linprog(c = obj, A_ub = lsh_ineq, b_ub = rsh_ineq, A_eq = lsh_eq, b_eq = rsh_eq,  method = "interior-point",options = {"sparse": False,"presolve": True, "tol": 1e-12})
    return opt.x, opt.fun

def linear_combination_labeler(theta,X):
    num_wls, num_classes = np.shape(X)
    cum = np.zeros(num_classes)

    for i in range(num_wls):
        cum = np.add(cum, np.multiply(X[i],theta[i]))
    return cum

def logistic_regression(theta, X):

    ret = np.exp(np.dot(X, theta))
    ret /= np.sum(ret)
    # ret = ret/cum
    return ret

def projectToSimplex(v):
    v = np.array(v)
    #Code from https://gist.github.com/mblondel/6f3b7aaad90606b98f71
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

def projectToBall(v, maxDist = 10000):
    v = np.array(v)
    c = min(100/np.linalg.norm(v),1)
    return v/c

def computeGradientLR(theta, X, Y, h):
    M = len(X)
    Y = np.array(Y)
    Y_predict = np.array([ h(theta, X[i]) for i in range(M)])
    Y_diff = Y_predict - Y
    grads = np.array([np.kron(Y_diff[i],X[i]) for i in range(M)])
    grad = np.average(grads,axis=0)
    return grad

def computeGradientCOMB(theta, X, Y, h):
    X = np.array(X)
    Y = np.array(Y)
    pred = np.array([ h(theta, X[:,j]) for j in range(len(Y))])
    grad = np.array([2*np.dot(X[:,j],(pred[j]-Y[j])) for j in range(len(Y))])
    grad = np.average(grad,axis=0)
    return grad

def subGradientMethod(X_unlabeled, constraint_matrix,constraint_vector,constraint_sign, lf, h, proj_function, initial_theta, iteration, step_size, N, M, C, lr=False):
    '''
    N - Number of weak classifiers
    M - Number of unlabeled data points
    C - Number of classes
    '''

    def eval_theta(th):
        cum = 0
        for j in range(M):
            cum += lf(new_y[j], h(th,X_unlabeled[:,j]))

        return cum/M
    
    def eval_lr(th):
        cum = 0
        for j in range(M):
            cum += lf(new_y[j], h(th,X_unlabeled[j]))
        return cum/M
    
    def eval_lr_iterative(th):
        cum = 0
        
        # for j in range(M):
            # cum += lf(new_y[j], h(th.reshape(-1, C),X_unlabeled[j]))
        # cum = np.sum(lf(new_y, h(th, X_unlabeled)))
        cum = np.sum(lf(new_y, h(th.reshape(-1, C), X_unlabeled)))
        return cum/M

    best_val = 10e10
    theta = initial_theta
    cost= []
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

    # START ITERATIONS
    for t in range(iteration):
        # Compute gradient with respect to theta
        EPS = 1e-5
        if lr:
            grad = computeGradientLR(theta,X_unlabeled,new_y,h)
            theta = theta.T.flatten()
        else:
            grad = computeGradientCOMB(theta, X_unlabeled, new_y,  h)


        # Gradient descent step
        theta = np.add(theta, np.multiply(grad,-step_size))
        # Projection step
        theta = proj_function(theta)

        if lr:
            # convert back to matrix form
            theta = theta.reshape((C, -1)).T
        
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
        # print(new_y)
        if not lr:
            current_eval = eval_theta(theta)
        if lr:
            current_eval = eval_lr(theta)
        # Update best result
        if (current_eval < best_val):
            best_theta = theta
            best_val = current_eval

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

    if not lr:
        print("ENDING PARAMETERS: " + str(best_theta))

	
    return best_theta