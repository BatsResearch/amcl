import numpy as np
import torch
from torchvision.models import resnet

###################################################################
# Loss Function Implementations
###################################################################

def Brier_loss_linear(labels, preds):
    '''
    Computing brier score given labels and predictions
    '''
    # Check if y_1 and y_2 have the same length
    if len(labels) != len(preds):
        raise NameError('Computing loss on vectors of different lengths')

    sq = np.sum(np.square(preds))
    tmp1 = -np.square(preds) + sq
    tmp2 = np.square(1.0 - preds)
    x = np.dot(labels,tmp1) + np.dot(labels,tmp2)
    return x / 2

def Brier_Score_AMCL(preds):
    '''
    Computing brier score given predictions against all possible labelings
    '''

    sq = np.sum(np.square(preds))
    tmp1 = -np.square(preds) + sq
    tmp2 = np.square(1.0 - preds)
    return (tmp1 + tmp2) / 2

def mse(y_1, y_2):
    '''
    Helper function for mean squared error
    '''
    return np.sum(np.square(y_1 - y_2)) / len(y_1)

def cross_entropy_linear(labels, predictions):
    '''
    Computing cross entropy given labels and predictions
    '''
    # Clipping values for numerical issues
    y_2 = np.clip(predictions, 1e-5, None)   
    return np.sum(-labels * np.log(y_2))

def resnet_transform(unlabeled_data):
	'''
	Function to transform unlabeled data into feature representation given by 
	pre-trained resnet (i.e. running AMCL on images)
	'''

	res = resnet.resnet18(pretrained=True)
	fr = res(torch.tensor(unlabeled_data)).detach().numpy()
	return fr

##################################
# Models implementation
##################################

def linear_combination_labeler(theta,X):
    '''
    Implementation of a linear combination (convex) of the weak classifiers.
    Note we restrict the sum of theta to be 1

    Args:
    theta - params, X - data
    '''

    num_wls, num_classes = np.shape(X)
    cum = np.zeros(num_classes)
    for i in range(num_wls):
        cum = np.add(cum, np.multiply(X[i],theta[i]))
    return cum

def logistic_regression(theta, X):
    '''
    Implementation of a logistic regression model

    Args:
    theta - params, X - data
    '''
    ret = np.exp(np.dot(X, theta))
    ret /= np.sum(ret)
    return ret

##################################################
# Projection methods used during gradient descent
##################################################


def projectToSimplex(v):
    '''
    Project a vector to the simplex
    Code - implementation taken from https://gist.github.com/mblondel/6f3b7aaad90606b98f71
    '''

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

def projectToSimplexLR(w):
    '''
    Projection of weights matrix to simplex
    '''

    n_features = w.shape[1]
    u = np.sort(w)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(w - theta, 0)
    return w

def projectToBall(v, maxDist = 100):
    '''
    Project a vector inside a sphere of radius maxDist
    '''
    v = np.array(v)
    c = min(maxDist/np.linalg.norm(v),1)
    return v * c

def projectCC(v):
    '''
    Project to valid convex combination (i.e. must sum to 1)
    '''

    return v / np.sum(v)

##################################
# Gradient computations
##################################

def computeGradientLR(theta, X, Y, h):
    '''
    Gradient computation for multinomial logistic regression with cross-entropy loss
    
    Args:
    X - data, Y - labels, h - logistic regression function (def. above)
    '''
    M = len(X)
    
    Y_predict = np.array([h(theta, X[i]) for i in range(M)])
    Y_diff = Y_predict - Y
    grads = np.array([np.kron(Y_diff[i], X[i]) for i in range(M)])

    # grads = np.array([np.outer(X[i], Y_diff[i]) for i in range(M)])
    grad = np.mean(grads, axis=0)
    return grad

def computeGradientCOMB(theta, X, Y, h):
    '''
    Gradient computation for convex combination with Brier Loss
    
    Args:
    X - data, Y - labels, h - convex combination function (def. above)
    '''
    X = np.array(X)
    Y = np.array(Y)
    pred = np.array([ h(theta, X[:,j]) for j in range(len(Y))])
    grad = np.array([2*np.dot(X[:,j],(pred[j]-Y[j])) for j in range(len(Y))])
    grad = np.average(grad,axis=0)
    return grad