'''
Model simulation from Heston model state space
'''
# pylint: disable-msg=C0103
# pylint: disable-msg=W0621


import numpy as np


def state_space_heston_discrete(param, prev_obs):
    '''
    State space heston discrete (need better definition)
    '''
    r = param["r"]
    k = param["k"]
    theta = param["theta"]
    sigma = param["sigma"]
    delta = param["delta"]
    rho = param["rho"]
    # delta_W_v = delta_motions[0]
    # delta_W_s = delta_motions[1]
    V_tilde = (prev_obs + k * (theta - max((prev_obs, 0))) * delta
               + sigma * np.sqrt(max((prev_obs, 0))) * np.sqrt(delta) * np.random.normal())
    V = max((V_tilde, 0))
    z = ((r - V/2) * delta
         + np.sqrt((1 - rho**2)*V) * np.sqrt(delta) * np.random.normal()
         + rho * np.sqrt(V) * np.sqrt(delta) * np.random.normal())
    return z, V, V_tilde


def generate_stock_data(n, param):
    '''
    Generating stock data
    '''
    z = np.empty(n + 1)
    V = np.empty(n + 1)
    V_tilde = np.empty(n + 1)
    S = np.empty(n + 2)
    V[0] = 0.2
    V_tilde[0] = 0.2
    S[1] = 100
    for k in range(1, n+1):
        s = state_space_heston_discrete(param, V_tilde[k-1])
        z[k] = s[0]
        V[k] = s[1]
        V_tilde[k] = s[2]
        S[k+1] = S[k] * np.exp(z[k])
    return S, z, V


if __name__ == '__main__':
    n = 1000
    param = {"r" : 0.005, "k" : 1, "theta" : 0.25, "sigma" : 0.5, "delta" : 0.01, "rho" : 0.0001}
    S, z, V = generate_stock_data(n, param)
