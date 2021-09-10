import numpy as np
from statistics import mean
from scipy.optimize import linprog
import random
from scipy.optimize import approx_fprime
from algorithms.util import Brier_loss_linear

def computeMeanAndCovariance(output_labelers_labeled,true_labels, loss):
    N = len(output_labelers_labeled)
    M = len(output_labelers_labeled[0])
    C = len(output_labelers_labeled[0][0])

    # Compute mean
    mu = []
    for i in range(N):
        cum = 0
        for j in range(M):
            cum+= loss(true_labels[j], output_labelers_labeled[i][j])
        cum = cum/M
        mu.append(cum)
    
    # Compute covariance
    K = np.zeros((N,N))
    for i1 in range(N):
        for i2 in range(N):
            cum = 0
            for j in range(M):
                cum+= (loss(true_labels[j], output_labelers_labeled[i1][j]) - mu[i1])*(loss(true_labels[j], output_labelers_labeled[i2][j]) - mu[i2])
            cum = cum/(M-1)
            K[i1][i2] = cum
    return mu,K

def test_matrix(output_labelers_labeled,true_labels,output_labelers_unlabeled, lf):
    mean,covariance = computeMeanAndCovariance(output_labelers_labeled,true_labels,lf)
    N = len(output_labelers_unlabeled)
    M = len(output_labelers_unlabeled[0])
    C = len(output_labelers_unlabeled[0][0])

    # Invert the covariance matrix
    L = np.linalg.cholesky(covariance)
    Li = np.linalg.inv(L)
    covariance = np.dot(Li.T,Li)

    # x first N components are the coordinate of the vector of the objective function, next M*C are auxiliary variables
    # See qopsolvers documentation for name of variables
    P_tot = np.zeros( (M*C, M*C))
    for a in range(N):
        for b in range(N):
            P = np.zeros( (M*C,M*C))
            for j1 in range(M):
                for j2 in range(M):
                    for c1 in range(C):
                        for c2 in range(C):
                            e_1 = np.zeros(C)
                            e_1[c1]=1
                            e_2 = np.zeros(C)
                            e_2[c2]=1
                            P[j1*C+c1][j2*C+c2] = (lf(e_1, output_labelers_unlabeled[a][j1]) - mean[a])*(lf(e_2,output_labelers_unlabeled[b][j2])-mean[b])/M
            P = P*covariance[a][b]
            P_tot = np.add(P_tot,P)
    P_tot = (P_tot + P_tot.T)/2

    L = np.zeros( (M*C,N))
    for j in range(M):
        for c in range(C):
            e = np.zeros(C)
            e[c] = 1
            for i in range(N):
                L[j*C + c][i] = (lf(e, output_labelers_unlabeled[i][j]) - mean[i])/np.sqrt(M)

    P_tot2 = L @ covariance @ L.T
    P_tot2 = (P_tot2 + P_tot2.T)/2
    print(P_tot)
    print(P_tot2)

def maximumLikelihood(output_labelers_labeled,true_labels,output_labelers_unlabeled, lf):
    mean,covariance = computeMeanAndCovariance(output_labelers_labeled,true_labels,lf)
    N = len(output_labelers_unlabeled)
    M = len(output_labelers_unlabeled[0])
    C = len(output_labelers_unlabeled[0][0])
    # x first N components are the coordinate of the vector of the objective function, next M*C are auxiliary variables
    # See qopsolvers documentation for name of variables

    # Invert the covariance matrix
    L = np.linalg.cholesky(covariance)
    Li = np.linalg.inv(L)
    covariance = np.dot(Li.T,Li)

    x = cp.Variable(N+M*C)


    A  = [] # Equality constraints
    b = [] # 
    for i in range(N):
        cnst = np.zeros(N+M*C)
        cnst[i] = -M
        for j in range(M):
            for c in range(C):
                e = np.zeros(C)
                e[c] = 1
                cnst[N+j*C+c] = lf(e,output_labelers_unlabeled[i][j]) - mean[i]
        A.append(cnst)
        b.append(0)

    for j in range(M):
        cnst = np.zeros(N+M*C)
        for c in range(C):
            cnst[N+j*C+c] = 1
        A.append(cnst)
        b.append(1)
    A=np.array(A)
    b=np.array(b)
    
    P = np.zeros((N+M*C,N+M*C))
    P[:N,:N] = covariance
    h = np.zeros( N+M*C)
    for i in range(N):
        h[i] = -9999999
    prob = cp.Problem(cp.Minimize(cp.quad_form(x,P)), [A @ x == b, x >= h])
    prob.solve(solver =cp.ECOS,verbose=True,max_iters = 10000, abstol_inacc = 1e-2)
    return prob.value, x.value

def maximumLikelihood2(output_labelers_labeled,true_labels,output_labelers_unlabeled, lf):
    mean,covariance = computeMeanAndCovariance(output_labelers_labeled,true_labels,lf)
    N = len(output_labelers_unlabeled)
    M = len(output_labelers_unlabeled[0])
    C = len(output_labelers_unlabeled[0][0])

    # Invert the covariance matrix
    L = np.linalg.cholesky(covariance)
    Li = np.linalg.inv(L)
    covariance = np.dot(Li.T,Li)

    L = np.zeros( (M*C,N))
    for j in range(M):
        for c in range(C):
            e = np.zeros(C)
            e[c] = 1
            for i in range(N):
                L[j*C + c][i] = (lf(e, output_labelers_unlabeled[i][j]) - mean[i])/np.sqrt(M)

    P_tot = L @ covariance @ L.T
    P_tot = (P_tot + P_tot.T)/2

    A  = [] # Equality constraints
    b = [] # 
    for j in range(M):
        cnst = np.zeros(M*C)
        for c in range(C):
            cnst[j*C+c] = 1
        A.append(cnst)
        b.append(1)
    A=np.array(A)
    b=np.array(b)
    x=cp.Variable(M*C)
    h=np.zeros(M*C)
    prob = cp.Problem(cp.Minimize(cp.quad_form(x,P_tot)), [A @ x == b, x >= h])
    prob.solve(solver =cp.ECOS,verbose=True,max_iters = 1000, abstol_inacc = 1e-2)
    return prob.value, x.value