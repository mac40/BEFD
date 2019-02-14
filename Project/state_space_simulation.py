'''
Model simulation from Heston model state space
'''
# pylint: disable-msg=C0103
# pylint: disable-msg=W0621


import numpy as np
import matplotlib.pyplot as plt


def state_space_heston_discrete_euler(param, prev_obs):
    '''
    Volatility simulation with euler discretisation
    '''
    r = param["r"]
    k = param["k"]
    theta = param["theta"]
    sigma = param["sigma"]
    delta = param["delta"]
    rho = param["rho"]
    epsilons = [np.random.normal(), np.random.normal()]
    V_tilde = (prev_obs + k * (theta - max((prev_obs, 0))) * delta
               + sigma * np.sqrt(max((prev_obs, 0))) * np.sqrt(delta) * epsilons[1])
    V = max((V_tilde, 0))
    z = ((r - V/2) * delta
         + np.sqrt((1 - rho**2)*V) * np.sqrt(delta) * epsilons[0]
         + rho * np.sqrt(V) * np.sqrt(delta) * epsilons[1])
    return z, V, V_tilde

def state_space_heston_discrete_nmle(param, prev_obs):
    '''
    Volatility simulation with euler discretisation
    '''
    r = param["r"]
    k = param["k"]
    theta = param["theta"]
    sigma = param["sigma"]
    delta = param["delta"]
    rho = param["rho"]
    epsilons = [np.random.normal(), np.random.normal()]
    eta = 0.0001
    V_tilde_sq = (np.sqrt(prev_obs)
                  + 1/(2 * np.sqrt(np.max(prev_obs, eta)))
                  * (k * theta - k * prev_obs - 1/4 * sigma ** 2) * delta
                  + 1/2 * sigma * np.sqrt(delta) * epsilons[1])
    V_sq = max((V_tilde_sq, 0))
    V = V_sq ** 2
    V_tilde = V_tilde_sq ** 2
    z = ((r - V/2) * delta
         + V_sq * np.sqrt(1 - rho**2) * np.sqrt(delta) * epsilons[0]
         + rho * V_sq * np.sqrt(delta) * epsilons[1])
    return z, V, V_tilde


def generate_stock_data(n, param, discretization):
    '''
    Generating stock data
    '''
    z = np.empty(n + 1)
    V = np.empty(n + 1)
    V_tilde = np.empty(n + 1)
    S = np.empty(n + 2)
    V[0] = 0.2
    V_tilde[0] = 0.2
    S[0] = 100
    S[1] = 100
    for k in range(1, n+1):
        if discretization == "euler":
            s = state_space_heston_discrete_euler(param, V_tilde[k-1])
        if discretization == "nmle":
            s = state_space_heston_discrete_euler(param, V_tilde[k-1])
        z[k] = s[0]
        V[k] = s[1]
        V_tilde[k] = s[2]
        S[k+1] = S[k] * np.exp(z[k])
    return S, z, V


if __name__ == '__main__':
    n = 1000
    param = {"r" : 0.005, "k" : 1, "theta" : 0.25, "sigma" : 0.5, "delta" : 0.01, "rho" : 0.0001}
    S, z, V = generate_stock_data(n, param, "nmle")
    plt.plot(S)
    plt.show()
