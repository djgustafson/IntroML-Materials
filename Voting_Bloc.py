#%%
import pandas as pd
import numpy as np
from scipy.special import digamma
from scipy.special import loggamma as lgamma

#%% Read in data
votes = pd.read_csv("/Users/danielgustafson/Dropbox/Variational Inference Presentation/votes.csv")

        
#%% r Update
def update_r(Lambda, eta, votes, n_sens, K):

    r = np.zeros((n_sens,K))

    for i in range(n_sens):
        for k in range(K):
            e_log_pi = digamma(Lambda[k]) - digamma(np.sum(Lambda)) 
            eta_sum = digamma(eta[0, :, k] + eta[1, :, k]) 
            e_log_theta_1 = digamma(eta[0, :, k]) - eta_sum
            e_log_theta_2 = digamma(eta[1, :, k]) - eta_sum 
            r[i,k] = e_log_pi + np.nansum(votes.iloc[i,] * e_log_theta_1 + (1 - votes.iloc[i,]) * e_log_theta_2)
        r[i, :] -= np.max(r[i, :]) #For numerical stability
        r[i,:] = np.exp(r[i,:])/np.sum(np.exp(r[i,:]))
    
    return(r)

#%% Lambda Update
def update_lambda(alpha, r, K):
    alphas = [alpha] * K
    Lambda = alphas + np.nansum(r, axis = 0)
    
    return(Lambda)


#%% Eta Update
def update_eta(r, gamma_1, gamma_2, votes, K, n_votes):
    
    eta = np.ndarray((2, n_votes, K))
    
    for k in range(K):
        r_i = r[:,k]
        for j in range(n_votes):
            eta[0, j, k] = gamma_1 + np.nansum(r_i * votes.iloc[:,j]) 
            eta[1, j, k] = gamma_2 + np.nansum(r_i * (1.0 - votes.iloc[:,j]))
    
    return(eta)
    
#%% Lower Bound
def lower_bound(gamma_1, gamma_2, alphas, Lambda, eta, r, votes, K, n_sens):
    
    gamma = np.array((gamma_1, gamma_2))
    res = lgamma(np.sum(gamma)) - np.sum(lgamma(gamma))
    res += lgamma(np.sum(alphas)) - np.sum(lgamma(alphas))
    lambda_sum = np.sum(Lambda)
    for k in range(K):
        dig_lambda = digamma(Lambda[k]) - digamma(lambda_sum)
        res += (alphas[k] - 1) * dig_lambda
        res -= (Lambda[k] - 1) * dig_lambda
        for j in range(K):
            eta_sum = np.sum(eta[:, j, k])
            for z in range(2):
                dig_eta = digamma(eta[z, j, k]) - digamma(eta_sum)
                res += (gamma[z] - 1) * dig_eta
                res -= (eta[z, j, k] - 1) * dig_eta
    for i in range(n_sens):
        for k in range(K):
            res += r[i,k] * (digamma(Lambda[k]) - digamma(lambda_sum))
            r_smooth = (r[i,k] + 1e-10)/np.sum(r[i,k] + 1e-10)
            res -= r_smooth * np.log(r_smooth) #For numerical stability
            eta_sum = eta[0, :, k] + eta[1, :, k]
            res += r[i,k] * np.nansum(votes.iloc[i, :] * (digamma(eta[0, :, k]) - digamma(eta_sum)) 
                + (1.0 - votes.iloc[i, :]) * (digamma(eta[1, :, k]) - digamma(eta_sum))) 
    
    return(res)

#%% Function to estimate
def Est_Vote_Bloc(K, alpha, gamma_1, gamma_2, votes):
    # Dimensions
    n_sens = votes.shape[0]
    n_votes = votes.shape[1]
    
    alphas = [alpha] * K
    
    # Initialize variational parameters
    Lambda = np.random.gamma(1, 1, K)
    eta = np.random.gamma(1, 1, (2, n_votes, K))
    
    r = update_r(Lambda, eta, votes, n_sens, K)
    Lambda = update_lambda(alpha, r, K)
    eta = update_eta(r, gamma_1, gamma_2, votes, K, n_votes)
    not_conv = True
    it = 0
    lbOld = lower_bound(gamma_1, gamma_2, alphas, Lambda, eta, r, votes, K, n_sens)
    tol = .00000001
    print("Estimating model...")
    while(not_conv & (it < 50)):
        r = update_r(Lambda, eta, votes, n_sens, K)
        Lambda = update_lambda(alpha, r, K)
        eta = update_eta(r, gamma_1, gamma_2, votes, K, n_votes)
        lbNew = lower_bound(gamma_1, gamma_2, alphas, Lambda, eta, r, votes, K, n_sens)
        if(it % 1 == 0):
          print(f"\tIter {it}: {lbNew:.3f}")
        if(np.abs(lbNew - lbOld) < tol):
          not_conv = False
        else:
          lbOld = lbNew 
          it += 1
    
    if(not_conv & (it > 50)):
        print(f"Warning: model did not converge after {it} iterations.")
    else:
        print(f"Model converged after {it} iterations.\nFinal LB: {lbNew:.3f}.")
        
#%% Estimate!
Est_Vote_Bloc(K=4, alpha=1, gamma_1=1, gamma_2=1, votes=votes)


