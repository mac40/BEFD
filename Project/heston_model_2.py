'''
Heston model implementation
'''
# pylint: disable-msg=C0103

import sys

import numpy as np
# import pandas as pd

import state_space_simulation as sss


def consistent_extended_kalman_filter(S, KAPTH_HAT, THETA_HAT,
                                      SIGMA_HAT, RO_HAT, deltaT, R, delta, Consistent=False):
    '''
    Estimation of volatility via CEKF
    '''
    # VARIABLES
    V_HAT = np.zeros(len(S))  # Predicted Volatility

    # AUXILIARY VARIABLES
    V_BAR = np.zeros(len(S))  # State prediction estimation
    P = np.zeros(len(S))  # Variance Covariance matrix upper-bound
    P_BAR = np.zeros(len(S))  # Prediction estimation-error covariance
    Q = [None]*len(S)  # list of 2x2 matrices
    K = np.zeros(len(S))  # Kalman Gain
    F = np.zeros(len(S))
    L = [None]*len(S)  # list of 2x1 matrices
    H = np.zeros(len(S))
    M = [None]*len(S)  # list of 2x1 matrices
    ZETA = np.zeros(len(S))

    # CONSISTENCY
    deltaQ = np.zeros(len(S))  # Consistency variable
    deltaR = np.zeros(len(S))  # Consistency variable

    # INITIALIZATION
    V_HAT[0] = 1
    P[0] = 0.1

    for i in range(1, len(S)):
        ZETA[i] = np.log(S[i]) - np.log(S[i-1])
    for i in range(0, len(S)):
        Q[i] = np.diag(np.array([1, 1]))

    # UPDATE
    for k in range(0, len(S)-1):

        # LINEARIZATION MATRICES OF THE STATE FUNCTION
        F[k] = 1 - KAPTH_HAT * deltaT
        L[k] = np.array([0, SIGMA_HAT * np.sqrt(V_HAT[k])])

        # STATE PREDICTION ESTIMATION AND PREDICTION ESTIMATION-ERROR COVARIANCE
        V_BAR[k+1] = (V_HAT[k]
                      + KAPTH_HAT * THETA_HAT * deltaT
                      - KAPTH_HAT * V_HAT[k] * deltaT)

        # DeltaQ for consistency
        if Consistent:
            deltaQ[k] = (P[k] * (1 - KAPTH_HAT * deltaT)**2
                         + deltaT**2 * (KAPTH_HAT * THETA_HAT)**2
                         + (SIGMA_HAT)**2 * deltaT * V_HAT[k] * Q[k][1, 1]
                         - F[k] * P[k] * np.transpose(F[k])
                         + L[k].dot(Q[k]).dot(np.transpose(L[k])))

        P_BAR[k+1] = (F[k] * P[k] * np.transpose(F[k])
                      + deltaQ[k])

        # LINEARIZATION MATRICES OF THE MEASUREMENT FUNCTION
        H[k+1] = -1/2 * deltaT
        M[k+1] = np.array([np.sqrt((1 - RO_HAT**2) * V_BAR[k+1]),
                           RO_HAT * np.sqrt(V_BAR[k+1])])

        # STATE ESTIMATE AND ERROR COVARIANCE
        K[k+1] = ((P_BAR[k+1] * np.transpose(H[k+1])
                   + L[k].dot(Q[k]).dot(np.transpose(M[k+1])))
                  * (H[k+1] * P_BAR[k] * np.transpose(H[k+1])
                     + M[k+1].dot(Q[k]).dot(np.transpose(M[k+1]))
                     + H[k+1] * L[k].dot(Q[k]).dot(np.transpose(M[k+1]))
                     + M[k+1].dot(Q[k]).dot(np.transpose(L[k])) * np.transpose(H[k+1]))**(-1))
        V_HAT[k+1] = (V_BAR[k+1] + K[k+1] * (ZETA[k+1] - ((R - 1/2 * V_BAR[k+1]) * delta)))

        # DeltaR for consistency
        if Consistent:
            deltaR[k+1] = (P_BAR[k+1] * (1 + (K[k+1] * deltaT)/2)**2
                           + 2 * K[k+1]**2 * deltaT * V_BAR[k]
                           * ((1-RO_HAT**2) * Q[k][0, 0]
                              + RO_HAT**2 * Q[k][1, 1]) - P_BAR[k+1]
                           + K[k+1] * (H[k+1] * P_BAR[k+1]
                                       + M[k+1].dot(np.transpose(L[k]))))

        P[k+1] = ((1 - K[k+1] * H[k+1]) * (P_BAR[k+1]) * (np.transpose(1 - K[k+1] * H[k+1]))
                  + K[k+1] * M[k+1].dot(Q[k+1]).dot(M[k+1].T) * K[k+1].T
                  - (1 - K[k+1] * H[k+1]) * (L[k]).dot(Q[k+1]).dot(M[k+1].T) * K[k+1].T
                  - K[k+1] * M[k+1].dot(Q[k+1]).dot(L[k].T) * ((1 - K[k+1] * H[k+1]).T)
                  + deltaR[k+1])

    # Save V_HAT for each iteration to external out file
    with open('./Project/v_hat.out', 'a') as outfile:
        aux_string = ""
        for item in V_HAT:
            aux_string = aux_string + " " + str(item)
        outfile.write("{}\n".format(aux_string))

    return V_HAT


def maximum_likelyhood_estimation(S, parameters, Consistent=False):
    '''
    Estimation of parameters via Maximum Likelihood Estimation\n
    with Extended Kalman Filter support for Volatility computation
    '''
    # ADDING S[0]
    S = np.append(np.array([1000]), S)  # new S[0]
    T = len(S)  # Number of measurements

    # VARIABLES
    V_HAT = np.zeros(T)
    KAPTH_HAT = np.zeros(T+1)
    THETA_HAT = np.zeros(T+1)
    SIGMA_HAT = np.zeros(T+1)
    RO_HAT = np.zeros(T+1)

    # AUXILIARY VARIABLES
    deltaW1 = np.zeros(T)
    deltaW2 = np.zeros(T)
    deltaT = parameters["delta"]
    delta = parameters["delta"]
    R = parameters["r"]  # Annual Interest Rate

    # INITIALIZATION
    KAPTH_HAT[0] = parameters["k"]
    THETA_HAT[0] = parameters["theta"]
    SIGMA_HAT[0] = parameters["sigma"]
    RO_HAT[0] = parameters["rho"]

    # UPDATE
    for i, _ in enumerate(S):
        # GENERATE VOLATILITIES FOR EACH MLE ITERATION
        V_HAT = consistent_extended_kalman_filter(
            S, KAPTH_HAT[i], THETA_HAT[i], SIGMA_HAT[i], RO_HAT[i], deltaT, R, delta, Consistent)

        # AUXILIARY VALUE FOR KAPTH, THETA, SIGMA GENERATION
        P_HAT = ((1/T * np.sum(np.sqrt(V_HAT[:T-1] * V_HAT[1:T]))
                  - 1/(T**2) * np.sum(np.sqrt(V_HAT[1:T]/V_HAT[:T-1])) * np.sum(V_HAT[:T-1]))
                 / (delta/2 - delta/2 * 1/(T**2) * np.sum(1/V_HAT[:T-1]) * np.sum(V_HAT[:T-1])))

        # KAPH THETA SIGMA GENERATION
        KAPTH_HAT[i+1] = (2/delta * (1 + (P_HAT * delta)/2 * 1/T * np.sum(1/V_HAT[:T-1])
                                     - 1/T * np.sum(np.sqrt(V_HAT[1:T]/V_HAT[:T-1]))))
        SIGMA_HAT[i+1] = np.sqrt(4/delta * 1/T
                                 * np.sum((np.sqrt(V_HAT[1:T])
                                           - np.sqrt(V_HAT[:T-1])
                                           - delta/(2 * np.sqrt(V_HAT[:T-1]))
                                           * (P_HAT - KAPTH_HAT[i+1] * V_HAT[:T-1]))**2))
        THETA_HAT[i+1] = (P_HAT + 1/4 * SIGMA_HAT[i+1]**2)/KAPTH_HAT[i+1]

        # AUXILIARY VARIABLES FOR RHO GENERATION
        for k in range(0, T-1):
            deltaW1[k+1] = (np.log(S[k+1]) - np.log(S[k]) -
                            (R - 1/2 * V_HAT[k]) * delta) / (np.sqrt(V_HAT[k]))
            deltaW2[k+1] = (V_HAT[k+1]
                            - V_HAT[k]
                            - KAPTH_HAT[i+1]
                            * (THETA_HAT[i+1] - V_HAT[k])
                            * delta) / (SIGMA_HAT[i+1] * np.sqrt(V_HAT[k]))

        # RHO GENERATION
        RO_HAT[i+1] = 1/(T * delta) * np.sum(deltaW1[1:T] * deltaW2[1:T])

    return V_HAT

if __name__ == "__main__":
    with open('./Project/v_hat.out', 'w') as out:
        out.write('')

    # REAL DATA
    # data = pd.read_csv(sys.argv[1])
    # data = np.array(data[-10000:].Close)

    # SIMULATED DATA
    param = {"r": 0.05, "k": 1, "theta": 0.05,
             "sigma": 0.01, "delta": 0.01, "rho": 0.01}
    data = sss.simulation(1000, param)[0]

    Volatility = maximum_likelyhood_estimation(data, param, sys.argv[2])
    print(Volatility)
