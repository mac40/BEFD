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
    V_tilde_sq = (prev_obs
                  + 1/(2 * np.sqrt(np.max(prev_obs ** 2, eta)))
                  * (k * theta - k * prev_obs ** 2 - 1/4 * sigma ** 2) * delta
                  + 1/2 * sigma * np.sqrt(delta) * epsilons[1])
    V_sq = max((V_tilde_sq, 0))
    V = V_sq ** 2
    z = ((r - V/2) * delta
         + np.sqrt((1 - rho**2)*V) * np.sqrt(delta) * epsilons[0]
         + rho * V_sq * np.sqrt(delta) * epsilons[1])
    return z, V, V_tilde_sq


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


def simulation(n, param):
    '''
    data simulation revisited
    '''
    V = np.zeros(n)
    V[0] = 0.05
    sqrt_V = np.zeros(n)
    sqrt_V[0] = np.sqrt(V[0])
    z = np.zeros(n)
    z[0] = 0
    S = np.zeros(n)
    S[0] = 1000
    Wv = np.random.normal(0, 1, n)
    Ws = np.random.normal(0, 1, n)
    kappa = param["k"]
    theta = param["theta"]
    sigma = param["sigma"]
    rho = param["rho"]
    r = param["r"]
    delta = param["delta"]

    for i in range(1, n):
        deltaWv = Wv[i] - Wv[i-1]
        deltaWs = Ws[i] - Ws[i-1]
        sqrt_V[i] = (sqrt_V[i-1] + (kappa * theta - kappa * sqrt_V[i-1]**2
                                    - (sigma**2)/4) * delta/2/sqrt_V[i-1] + sigma * deltaWv/2)
        V[i] = V[i-1] + kappa * (theta - V[i-1]) * delta + sigma * V[i-1]**(1/2) * deltaWv
        z[i] = (r-V[i]/2) * delta + ((1-rho**2) * V[i])**0.5 * deltaWs + rho * V[i]**0.5 * deltaWv
        S[i] = S[i-1] * np.exp(z[i])
    return S, z, V, sqrt_V

def nmle(V_HAT,delta,R):
    
    T = len(V_HAT)
    deltaW1 = np.zeros(T)
    deltaW2 = np.zeros(T)
    
    P_HAT = ((1/T * np.sum(np.sqrt(V_HAT[:T-1] * V_HAT[1:T]))
                  - 1/(T**2) * np.sum(np.sqrt(V_HAT[1:T]/V_HAT[:T-1])) * np.sum(V_HAT[:T-1]))
                 / (delta/2 - delta/2 * 1/(T**2) * np.sum(1/V_HAT[:T-1]) * np.sum(V_HAT[:T-1])))

    # KAPH THETA SIGMA GENERATION
    KAPTH_HAT = (2/delta * (1 + (P_HAT * delta)/2 * 1/T * np.sum(1/V_HAT[:T-1]) 
                        - 1/T * np.sum(np.sqrt(V_HAT[1:T]/V_HAT[:T-1]))))
    
    SIGMA_HAT = np.sqrt(4/delta * 1/T
                             * np.sum((np.sqrt(V_HAT[1:T])
                                       - np.sqrt(V_HAT[:T-1])
                                       - delta/(2 * np.sqrt(V_HAT[:T-1]))
                                       * (P_HAT - KAPTH_HAT * V_HAT[:T-1]))**2))
    THETA_HAT = (P_HAT + 1/4 * SIGMA_HAT**2)/KAPTH_HAT
    
    for k in range(0, T-1):
            deltaW1[k+1] = (np.log(S[k+1]) - np.log(S[k]) -
                            (R - 1/2 * V_HAT[k]) * delta) / (np.sqrt(V_HAT[k]))
            deltaW2[k+1] = (V_HAT[k+1]
                            - V_HAT[k]
                            - KAPTH_HAT
                            * (THETA_HAT - V_HAT[k])
                            * delta) / (SIGMA_HAT * np.sqrt(V_HAT[k]))

    # RHO GENERATION
    RO_HAT = 1/(T * delta) * np.sum(deltaW1[1:T] * deltaW2[1:T])
    
    return KAPTH_HAT, SIGMA_HAT,THETA_HAT,RO_HAT
        

if __name__ == '__main__':
    # initialization parameters
    n = 3000
    param = {"r": 0.005, "k": 1, "theta": 0.25,
             "sigma": 0.5, "delta": 0.01, "rho": 0.001}

    # simulation
    S, z, V = generate_stock_data(n, param,'nmle')

    # Plotting generated data
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].plot(S)
    axes[0, 0].set_title('S')
    axes[0, 1].plot(z)
    axes[0, 1].set_title('z')
    axes[1, 0].plot(V)
    axes[1, 0].set_title('V')
    """    axes[1, 1].plot(sqrt_V)
    axes[1, 1].set_title('sqrt_V')"""
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()
    
    KAPTH_HAT, SIGMA_HAT,THETA_HAT,RO_HAT = nmle(V,delta = 0.01, R = 0.005)
    print(KAPTH_HAT, SIGMA_HAT,THETA_HAT,RO_HAT)