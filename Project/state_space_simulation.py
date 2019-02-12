'''
Model simulation from Heston model state space
'''
# pylint: disable-msg=C0103
# pylint: disable-msg=W0621


import numpy as np
import pandas as pd


def generate_brownian_motion(n, delta_t=1):
    '''
    Function to generate brownian motion
    '''
    B = np.empty(n+1)
    B[0] = 0
    for i in range(1, n+1):
        B[i] = np.random.normal(0, delta_t) + B[i-1]
    return B


def state_space_heston_discrete(param, prev_obs, delta_motions):
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


def generate_stock_data(n, param, motions):
    '''
    Generating stock data
    '''
    W_v = motions.W_v
    W_s = motions.W_s
    delta_W_v = np.diff(W_v)
    delta_W_s = np.diff(W_s)
    z = np.empty(n + 1)
    V = np.empty(n + 1)
    V_tilde = np.empty(n + 1)
    S = np.empty(n + 2)
    V[0] = 0.2
    V_tilde[0] = 0.2
    S[1] = 100
    for k in range(1, n+1):
        s = state_space_heston_discrete(param, V_tilde[k-1], [delta_W_v[k-1], delta_W_s[k-1]])
        z[k] = s[0]
        V[k] = s[1]
        V_tilde[k] = s[2]
        S[k+1] = S[k] * np.exp(z[k])
    return S, z, V


if __name__ == '__main__':
    n = 1000
    W_v = generate_brownian_motion(n).reshape(n+1, 1)
    W_s = generate_brownian_motion(n).reshape(n+1, 1)
    motions = pd.DataFrame(np.concatenate([W_v, W_s], axis=1), columns=["W_v", "W_s"])
    param = {"r" : 0.005, "k" : 1, "theta" : 0.25, "sigma" : 0.5, "delta" : 0.01, "rho" : 0.0001}
    S, z, V = generate_stock_data(n, param, motions)
