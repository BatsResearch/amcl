import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
from sklearn.linear_model import LogisticRegression
from algorithms.util import Brier_loss_linear, Brier_Score_AMCL, cross_entropy_linear
from algorithms.util import computeGradientLR, computeGradientCOMB
##################################

def compute_constraints_with_loss(lf, output_labelers_unlabeled, output_labelers_labeled, true_labels): 
    
    '''
    This function builds the linear constraints that represents the
    feasible set of labeling based on the unlabeled data
    and the labeled data. The output of this functions are three
    elements: constraint_matrix, constraint_vector, constraint_sign
    The i-th element of each of these three vectors is respectively
    the coefficients, the constant term, and the kind of the inequality
    ( <=, ==, >=) of the i-th linear constraint of the feasible set.
    The variables of these linear constraints are M*C, where M
    is the number of unlabeled data points and C is the number
    of classes. The (j*C + c)-th variable represents the probability
    that the j-th unlabeled item belongs to class c.

    Args:
    lf - loss function (as defined above)
    output_labelers_unlabeled - votes of weak supervision on unlabeled data
    output_labelers_labeled - votes of weak supervision on labeled data
    true_labels - labels of labeled data
    '''

    constraint_matrix = [] # The i-th element is a vector that represents the coefficient of the i-th linear constraint
    constraint_vector = [] # The i-th element is the constant term of the i-th linear constraint
    constraint_sign = [] # The i-th element represents the sign of the inequality of the i-th linear constraint
                         # i.e., -1 is <=, 0 is ==, 1 is >=

    N = len(output_labelers_unlabeled) # Number of weak classifiers
    M = len(output_labelers_unlabeled[0]) # Number of unlabeled data points
    C = len(output_labelers_unlabeled[0][0]) # Number of classes
    Ml = len(output_labelers_labeled[0]) # Number of labeled data points
    print("Num WL: %d, Num Unlab %d, Num Classes %d, Num Lab %d" % (N,M,C,Ml))

    if Ml != len(true_labels):
        raise NameError('Labeled data points and label sizes are different')

    # For each weak classifier
    for i in range(N):
        
        error = 0
        for j in range(Ml):
            error += lf(true_labels[j],output_labelers_labeled[i][j])
        error = error/Ml

        # Compute the coefficient of the linear constraint based on the 
        build_coefficients = np.zeros((C * M))
        one_hots = np.eye(C)
        for j in range(M):
            for c in range(C):
                build_coefficients[C * j + c] = lf(one_hots[c], output_labelers_unlabeled[i][j])/M # For later, check the division here

        # Bounds: risk of a labeler must be within error+-offset
        delta = 0.1 # Between 0 and 1. Probability of the true labeling to NOT belong to the feasible set.
        scaling_factor = 0.1 # Direct computation of the offset could yield large values if M or Ml is small.
                             # This number can be used to scale the offset if it is too large 
        offset = scaling_factor * np.sqrt((Ml + M)*np.log(4*N/delta)/(2*(Ml*M)))
        offset = 0 # Uncomment this line 
                   # if you do not want to have a offset. This could be better in practice if
                   # the number of labeled data and labeled data is very large

        # Add less or equal constraint
        constraint_matrix.append(build_coefficients)
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
def solveLPGivenCost(constraint_matrix,constraint_vector,constraint_sign,cost_vector):

    '''
    This function computes the worst-case error on the unlabeled data of a 
    classifier represented with cost_vector with respect to an adversarial choice
    of the feasible labeling of the unlabeled data (i.e., it solves the max of the minimax).
    cost_vector[j*C + c] is the error of the classifier on the j-th item if its true label was c.

    Args:
    first three - outputs of function above
    cost_vector - associated cost with particular labelings
    '''

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
            lsh_ineq.append( np.multiply(constraint_matrix[i], - constraint_sign[i]))
            rsh_ineq.append( np.multiply(constraint_vector[i], - constraint_sign[i]))

    # Solve the linear program
    opt = linprog(c = obj, A_ub = lsh_ineq, b_ub = rsh_ineq, A_eq = lsh_eq, b_eq = rsh_eq, 
                  method = "highs")
    return opt.x, opt.fun

##################################


def subGradientMethod(X_unlabeled, constraint_matrix,constraint_vector,constraint_sign, 
    lf, h, proj_function, initial_theta, iteration, step_size, N, M, C, lr=False):

    '''
    Running the subgradient method (via LP with Scipy optimize library)

    Args:
    * X_unlabeled => unlabeled data (M, *) - i.e. for votes it is (M, N, C)
    * constraint_matrix, constraint_vector, constraint_sign => constraints of the feasible set
      as computed from the method compute_constraints_with_loss
    * lf => loss function
    * h => prediction model (the first argument is the weights of the model)
    * proj_function => projection method for the weights of the prediction model
    * initial_theta => initial weights of the prediction model
    * iteration => number of iterations
    * step_size => step size for the subgradient descent method
    * N => number of weak classifiers
    * M => number of unlabeled data points
    * C => number of classes
    * lr => flag: true if we are using multinomial logistic regression - false if we are using
      convex combination of weak classifiers
    '''

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


    # Current value of the minimax
    best_val = 10e10 # Initialized to a very high value
    theta = initial_theta # Weights of the model
    best_theta = initial_theta.copy()

    if lr:
        preds = np.array([h(theta, X_unlabeled[i, :]) for i in range(M)])
    else:
        preds = np.array([h(theta, X_unlabeled[:,i]) for i in range(M)])
        
    # cost2 = Brier_Score_AMCL(preds).flatten() * -1 / M
    one_hots = np.eye(C)
    cost = np.zeros(M * C)
    for j in range(M):
        for c in range(C):
            ind = j * C + c
            if lr:
                cost[ind] = (-lf(one_hots[c], h(theta, X_unlabeled[j])) / M)
            else:
                cost[ind] = (-lf(one_hots[c], h(theta,X_unlabeled[:,j])) / M)

    # Find labeling that maximizes the error
    new_y,_ = solveLPGivenCost(constraint_matrix,constraint_vector,constraint_sign,cost)
    new_y = np.array([new_y[i*C : (i + 1) * C] for i in range(M)])

    if not lr:
        print("STARTING PARAMETERS: " + str(theta))

    # Subgradient method core implementation
    for t in range(iteration):
        # Compute subgradient with respect to theta and the current labeling of the unlabeled data
        if lr:
            grad = computeGradientLR(theta,X_unlabeled,new_y,h)
            theta = theta.T.flatten()
        else:
            grad = computeGradientCOMB(theta, X_unlabeled, new_y,  h)

        # Gradient descent step
        theta -= grad * step_size
        theta = proj_function(theta)

        # If multinomial logistic regression, convert weights in matrix form
        if lr:
            # convert back to matrix form
            theta = theta.reshape((C, -1)).T
        
        if lr:
            preds = np.array([h(theta, X_unlabeled[i, :]) for i in range(M)])
        else:
            preds = np.array([h(theta, X_unlabeled[:,i]) for i in range(M)])

        cost = np.zeros(M * C)
        for j in range(M):
            for c in range(C):
                ind = j * C + c
                if lr:
                    cost[ind] = (-lf(one_hots[c], h(theta, X_unlabeled[j])) / M)
                else:
                    cost[ind] = (-lf(one_hots[c], h(theta,X_unlabeled[:,j])) / M)

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
            best_theta = theta.copy()
            best_val = current_eval
        else:
            pass

        # Debug lines
        if t % 100 == 0:

            vals = []
            if lr:
                preds = np.array([h(theta, X_unlabeled[i]) for i in range(M)])     
                vals = np.array([lf(new_y[i], preds[i]) for i in range(M)])
            else:
                # constructing predictions
                totals = np.zeros((M, C))
                for i, val in enumerate(best_theta):
                    for j, vote in enumerate(X_unlabeled[i]):
                        totals[j] += val * vote
                for i in range(len(new_y)): 
                        vals.append(Brier_loss_linear(new_y[i], totals[i]))
            
            print("Bound at time %d: %f" % (t, np.mean(vals)))
            if not lr:
                print("Best Params:", best_theta)

    if not lr:
        print("ENDING PARAMETERS: " + str(best_theta))

    return best_theta

########################
# CVXPY implementation 
########################

def compute_constraints_with_loss2( lf1, lf2, output_labelers_unlabeled, output_labelers_labeled, true_labels, lr=False):

    '''
    Generating constraints for CVXPY implementation

    Args:
    lf1 - loss function (regular w/ two arguments)
    lf2 - loss function for adversarial labelling 
    output_labelers_unlabeled - votes of weak supervision on unlabeled data
    output_labelers_labeled - votes of weak supervision on labeled data
    true_labels - labels of labeled data
    '''

    N = len(output_labelers_unlabeled) # Number of weak classifiers
    M = len(output_labelers_unlabeled[0]) # Number of unlabeled data points
    C = len(output_labelers_unlabeled[0][0]) # Number of classes
    Ml = len(output_labelers_labeled[0]) # Number of labeled data points

    print("Num WL: %d, Num Unlab %d, Num Classes %d, Num Lab %d" % (N,M,C,Ml))

    # Bounds: risk of a labeler must be within error+-offset
    delta = 0.1 # Between 0 and 1. Probability of the true labeling to NOT belong to the feasible set. 
    B = 1  # Size of the range of the loss function 
    scaling_factor = 0.1 # Direct computation of the offset could yield large values if M or Ml is small.
                         # This number can be used to scale the offset if it is too large
    offset = B * scaling_factor * np.sqrt( (Ml + M)*np.log(4*N/delta)/(2*(Ml*M)))
    offset = 0 # Uncomment this line 
            # if you do not want to have a offset. This could be better in practice if
            #  the number of labeled data and labeled data is very large
    
    if Ml != len(true_labels):
        raise NameError('Labeled data points and label sizes are different')

    # Variables
    Y = cp.Variable((M,C),nonneg=True)

    # Constraint vector
    constraints = []

    # Add constraints for the sum of the label probabilities for each item to sum to 1
    for j in range(M):
        constraints.append(cp.sum(Y[j])==1)

    # Build constraint for weak classifier
    for i in range(N):
        # Compute the expected error over the labeled data for each weak classifier
        error = 0
        for j in range(Ml):
            error += lf1(true_labels[j],output_labelers_labeled[i][j])
            # print(error)
        error = error/Ml

        # Compute the coefficient of the linear constraint based on the 
        build_coefficients = np.zeros((M,C))
        for j in range(M):
            build_coefficients[j] = lf2(output_labelers_unlabeled[i][j])/M

        if(offset != 0):
            constraints.append( cp.sum(cp.multiply(Y,build_coefficients ) ) <= error + offset)
            constraints.append( cp.sum(cp.multiply(Y,build_coefficients ) ) >= error - offset)
        else:
            constraints.append( cp.sum(cp.multiply(Y,build_coefficients ) ) == error )


    # Add constraints for the sum of the label probabilities for each item to sum to 1
    return Y, np.array(constraints)


def solveLPGivenCost2(Y,constraints,cost):
    Y = Y.flatten()
    obj = cp.Minimize(cp.sum(cp.multiply(Y, cost)))
    prob = cp.Problem(obj, constraints)
    # prob.solve(solver=cp.ECOS, eps=1e-8)
    prob.solve(solver=cp.ECOS)
    # return np.reshape(Y, -1), prob.value
    return np.reshape(Y.value, -1), prob.value

def subGradientMethod2(X_unlabeled, Y, constraints, lf, h, proj_function, initial_theta, iteration, step_size, N, M, C, lr=False):

    '''
    Running the subgradient method (via LP with cvxpy)

    Args:
    * X_unlabeled => unlabeled data (M, *) - i.e. for votes it is (M, N, C)
    * Y => cvxpy variables for LP
    * constraints => constraints of the feasible set
      as computed from the method compute_constraints_with_loss
    * lf => loss function
    * h => prediction model (the first argument is the weights of the model)
    * proj_function => projection method for the weights of the prediction model
    * initial_theta => initial weights of the prediction model
    * iteration => number of iterations
    * step_size => step size for the subgradient descent method
    * N => number of weak classifiers
    * M => number of unlabeled data points
    * C => number of classes
    * lr => flag: true if we are using multinomial logistic regression - false if we are using
      convex combination of weak classifiers
    '''

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

    # Current value of the minimax
    best_val = 10e10 # Initialized to a very high value
    theta = initial_theta # Weights of the model

    if lr:
        preds = np.array([h(theta, X_unlabeled[i, :]) for i in range(M)])
    else:
        preds = np.array([h(theta, X_unlabeled[:,i]) for i in range(M)])

    one_hots = np.eye(C)
    cost = np.zeros(M * C)
    for j in range(M):
        for c in range(C):
            ind = j * C + c
            if lr:
                cost[ind] = (-lf(one_hots[c], h(theta, X_unlabeled[j])) / M)
            else:
                cost[ind] = (-lf(one_hots[c], h(theta,X_unlabeled[:,j])) / M)

    # Find labeling that maximizes the error
    new_y,_ = solveLPGivenCost2(Y, constraints, cost)
    new_y = np.array([new_y[i*C : (i + 1) * C] for i in range(M)])

    # Subgradient method core implementation
    for t in range(iteration):

        # Compute subgradient with respect to theta and the current labeling of the unlabeled data
        if lr:
            grad = computeGradientLR(theta, X_unlabeled, new_y, h)
            theta = theta.T.flatten()
        else:
            grad = computeGradientCOMB(theta, X_unlabeled, new_y,  h)

        # Gradient descent step
        theta -= grad * step_size
        # Projection step
        theta = proj_function(theta)

        # If multinomial logistic regression, convert weights in matrix form
        if lr:
            # convert back to matrix form
            theta = theta.reshape((C, -1)).T
            
        if lr:
            preds = np.array([h(theta, X_unlabeled[i]) for i in range(M)])
        else:
            preds = np.array([h(theta, X_unlabeled[:,i]) for i in range(M)])     

        cost = np.zeros(M * C)
        for j in range(M):
            for c in range(C):
                ind = j * C + c
                if lr:
                    cost[ind] = (-lf(one_hots[c], h(theta, X_unlabeled[j])) / M)
                else:
                    cost[ind] = (-lf(one_hots[c], h(theta,X_unlabeled[:,j])) / M)
        # Find labeling that maximizes the error
        new_y, obj = solveLPGivenCost2(Y, constraints, cost)
        new_y = np.array([new_y[i*C : (i + 1) * C] for i in range(M)])

        # Evaluate the current model with respect to the worst-case error
        if not lr:
            current_eval = eval_theta(theta)
        if lr:
            current_eval = eval_lr(theta)
        # If the current model is better, update the best model found
        if (current_eval < best_val):
            best_theta = theta.copy()
            best_val = current_eval

        # Debug lines
        if t % 10 == 0:

            vals = []
            if lr:
                preds = np.array([h(theta, X_unlabeled[i]) for i in range(M)])     
                vals = np.array([lf(new_y[i], preds[i]) for i in range(M)])
            else:
                # constructing predictions
                totals = np.zeros((M, C))
                for i, val in enumerate(best_theta):
                    for j, vote in enumerate(X_unlabeled[i]):
                        totals[j] += val * vote
                for i in range(len(new_y)): 
                        vals.append(Brier_loss_linear(new_y[i], totals[i]))
            
            print("Bound at time %d: %f" % (t, np.mean(vals)))
            if not lr:
                print("Best Params:", best_theta)

    # Debug lines
    if not lr:
        print("ENDING PARAMETERS: " + str(best_theta))

    # Return the best model found during the subgradient method execution
    return best_theta


############################
# subgradient method for LR
############################

def subGradientMethodLR(X_unlabeled, constraint_matrix,constraint_vector,constraint_sign, 
    lf, h, proj_function, initial_theta, iteration, step_size, N, M, C, lr=False):

    '''
    Running the subgradient method for logistic regression (w/ pre-training*)

    Args:
    * X_unlabeled => unlabeled data (M, *) - i.e. for votes it is (M, N, C)
    * constraint_matrix, constraint_vector, constraint_sign => constraints of the feasible set
      as computed from the method compute_constraints_with_loss
    * lf => loss function
    * h => prediction model (the first argument is the weights of the model)
    * proj_function => projection method for the weights of the prediction model
    * initial_theta => initial weights of the prediction model
    * iteration => number of iterations
    * step_size => step size for the subgradient descent method
    * N => number of weak classifiers
    * M => number of unlabeled data points
    * C => number of classes
    * lr => flag: true if we are using multinomial logistic regression - false if we are using
      convex combination of weak classifiers
    '''

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


    # Current value of the minimax
    best_val = 10e10 # Initialized to a very high value
    theta = initial_theta # Weights of the model
    best_theta = initial_theta.copy()

    preds = np.array([h(theta, X_unlabeled[i, :]) for i in range(M)])
    one_hots = np.eye(C)
    cost = np.zeros(M * C)
    for j in range(M):
        for c in range(C):
            ind = j * C + c
            cost[ind] = (-lf(one_hots[c], h(theta, X_unlabeled[j])) / M)

    # fully pre-training on valid point
    new_y,_ = solveLPGivenCost(constraint_matrix,constraint_vector,constraint_sign,cost)
    new_y = np.array([new_y[i*C : (i + 1) * C] for i in range(M)])

    lr_pretrain = LogisticRegression(penalty="l2")    
    labs = np.argmax(new_y, axis=1)
    lr_pretrain.fit(X_unlabeled, labs)
    
    for list_ind, lab_ind in enumerate(np.unique(labs)):
        theta[:, lab_ind] = lr_pretrain.coef_.T[:, list_ind]

    # Find labeling that maximizes the error
    new_y,_ = solveLPGivenCost(constraint_matrix,constraint_vector,constraint_sign,cost)
    new_y = np.array([new_y[i*C : (i + 1) * C] for i in range(M)])

    preds = np.array([h(theta, X_unlabeled[i, :]) for i in range(M)])
    one_hots = np.eye(C)
    cost = np.zeros(M * C)
    for j in range(M):
        for c in range(C):
            ind = j * C + c
            cost[ind] = (-lf(one_hots[c], h(theta, X_unlabeled[j])) / M)

    # Subgradient method core implementation
    for t in range(iteration):
        # Compute subgradient with respect to theta and the current labeling of the unlabeled data
        grad = computeGradientLR(theta,X_unlabeled,new_y,h)
        theta = theta.T.flatten()

        if t == 50 or t == 100:
            step_size = step_size / 2

        # Gradient descent step
        theta -= grad * step_size
        theta = proj_function(theta)

        # If multinomial logistic regression, convert weights in matrix form
        theta = theta.reshape((C, -1)).T
        preds = np.array([h(theta, X_unlabeled[i, :]) for i in range(M)])

        cost = np.zeros(M * C)
        for j in range(M):
            for c in range(C):
                ind = j * C + c
                cost[ind] = (-lf(one_hots[c], h(theta, X_unlabeled[j])) / M)

        # Find labeling that maximizes the error
        new_y, obj = solveLPGivenCost(constraint_matrix,constraint_vector,constraint_sign,cost)
        new_y = np.array([new_y[i*C : (i + 1) * C] for i in range(M)])

        # Evaluate the current model with respect to the worst-case error
        current_eval = eval_lr(theta)
        # If the current model is better, update the best model found
        if (current_eval < best_val):
            best_theta = theta.copy()
            best_val = current_eval

        # Debug lines
        if t % 10 == 0:
            vals = []
            preds = np.array([h(theta, X_unlabeled[i]) for i in range(M)])     
            vals = np.array([lf(new_y[i], preds[i]) for i in range(M)])
            print("Bound at time %d: %f" % (t, np.mean(vals)))

    return best_theta
